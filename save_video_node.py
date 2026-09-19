import json
import os
import subprocess
import tempfile

import numpy as np
import torch

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import InputImpl, io, ui
from comfy_extras.nodes_video import SaveVideo


class SaveVideoFast(SaveVideo):
    """Drop-in replacement for the core Save Video node that encodes with ffmpeg."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "SaveVideoFast"
        schema.display_name = "Save Video Fast"
        schema.description = (
            "Saves the input video to your ComfyUI output directory using a fast "
            "ffmpeg encode path. The running workflow is embedded in the MP4 metadata."
        )
        schema.inputs = [
            io.Video.Input("video", tooltip="The video to save."),
            io.String.Input(
                "filename_prefix",
                default="video/ComfyUI",
                tooltip=(
                    "The prefix for the file to save. This may include formatting information "
                    "such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."
                ),
            ),
            io.Combo.Input(
                "codec",
                options=["h264_nvenc", "h264"],
                default="h264_nvenc",
                tooltip="h264_nvenc uses NVIDIA GPU (fastest). h264 uses CPU (slower, but works without GPU).",
            ),
            io.Combo.Input(
                "preset",
                options=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                default="p1",
                tooltip="p1 = fastest encode (lowest quality). p7 = slowest (best quality). p1 is recommended for speed.",
            ),
            io.Int.Input(
                "video_quality",
                default=23,
                min=2,
                max=31,
                step=1,
                tooltip="Lower = better quality (higher bitrate). 23 is a good balance. Range: 2 (near lossless) - 31 (very low).",
            ),
        ]
        return schema

    @classmethod
    def execute(cls, video, filename_prefix, codec, preset, video_quality) -> io.NodeOutput:
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        file = f"{filename}_{counter:05}_.mp4"
        out_path = os.path.join(full_output_folder, file)

        metadata = {}
        if not args.disable_metadata:
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt

        if isinstance(video, InputImpl.VideoFromFile):
            cls._encode_file(video, out_path, metadata)
        else:
            cls._encode_components(video.get_components(), out_path, codec, preset, video_quality, metadata)

        return io.NodeOutput(
            video,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )

    @staticmethod
    def _metadata_args(metadata):
        cmd = []
        for key, value in metadata.items():
            cmd += ["-metadata", f"{key}={value if isinstance(value, str) else json.dumps(value)}"]
        return cmd

    @staticmethod
    def _encoder_args(codec, preset, quality):
        if codec == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", preset, "-cq", str(quality)]
        return ["-c:v", "libx264", "-crf", str(quality)]

    @classmethod
    def _encode_file(cls, video, out_path, metadata):
        source = video.get_stream_source()
        temp_source = None
        if not isinstance(source, (str, os.PathLike)):
            temp_source = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_source.write(source.read())
            temp_source.close()
            source = temp_source.name

        cmd = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", os.fspath(source)]
        cmd += ["-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-map_metadata", "0"]
        cmd += cls._metadata_args(metadata)
        cmd += ["-movflags", "+faststart+use_metadata_tags", out_path]

        try:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        finally:
            if temp_source is not None:
                try:
                    os.unlink(temp_source.name)
                except OSError:
                    pass

        if proc.returncode != 0:
            raise RuntimeError(f"[SaveVideoFast] ffmpeg failed:\n{proc.stderr.decode(errors='replace')}")

    @classmethod
    def _encode_components(cls, components, out_path, codec, preset, quality, metadata):
        frames = components.images
        if frames is None or len(frames) == 0:
            raise ValueError("[SaveVideoFast] No frames provided.")

        H, W = frames.shape[1], frames.shape[2]
        frame_rate = float(components.frame_rate)

        audio = components.audio
        audio_temp = None
        cmd = ["ffmpeg", "-y", "-loglevel", "quiet"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(frame_rate), "-i", "pipe:0"]

        if audio is not None and "waveform" in audio and "sample_rate" in audio:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if isinstance(waveform, torch.Tensor):
                wav_np = waveform.detach().cpu().numpy()
            else:
                wav_np = np.array(waveform)

            if wav_np.ndim == 1:
                wav_np = wav_np.reshape(1, -1)
            elif wav_np.ndim > 2:
                wav_np = wav_np.reshape(-1, wav_np.shape[-1])

            wav_int16 = (wav_np * 32767).astype(np.int16)
            if wav_int16.shape[0] > 1:
                wav_int16 = wav_int16.transpose(1, 0).reshape(-1)
            else:
                wav_int16 = wav_int16.flatten()

            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
            audio_temp.write(wav_int16.tobytes())
            audio_temp.close()

            cmd += [
                "-f", "s16le",
                "-ar", str(sample_rate),
                "-ac", str(wav_np.shape[0]),
                "-i", audio_temp.name,
                "-map", "0:v", "-map", "1:a", "-c:a", "aac",
            ]
        else:
            cmd += ["-map", "0:v"]

        cmd += cls._encoder_args(codec, preset, quality)
        cmd += cls._metadata_args(metadata)
        cmd += ["-movflags", "+faststart+use_metadata_tags", "-pix_fmt", "yuv420p", out_path]

        stderr_file = tempfile.TemporaryFile()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            stdin = proc.stdin
            assert stdin is not None
            try:
                if frames.is_cuda:
                    if frames.dtype == torch.uint8:
                        frames_cpu = frames.cpu(non_blocking=True)
                    else:
                        frames_cpu = (frames * 255).byte().contiguous().cpu(non_blocking=True)
                    torch.cuda.synchronize()
                else:
                    if frames.dtype == torch.uint8:
                        frames_cpu = frames.cpu()
                    else:
                        frames_cpu = (frames * 255).byte().contiguous().cpu()

                frames_np = frames_cpu.numpy()
                if frames_np.shape[-1] == 4:
                    frames_np = frames_np[:, :, :, :3]
                frames_np = np.ascontiguousarray(frames_np)
                data = frames_np.tobytes()

                CHUNK_SIZE = 512 * 1024 * 1024
                total_bytes = len(data)
                written = 0
                while written < total_bytes:
                    chunk = data[written:written + CHUNK_SIZE]
                    stdin.write(chunk)
                    written += len(chunk)
            except BrokenPipeError:
                pass
            finally:
                try:
                    stdin.close()
                except OSError:
                    pass

            proc.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
        finally:
            stderr_file.close()
            if audio_temp is not None:
                try:
                    os.unlink(audio_temp.name)
                except OSError:
                    pass

        if proc.returncode != 0:
            raise RuntimeError(f"[SaveVideoFast] ffmpeg failed:\n{stderr.decode(errors='replace')}")
