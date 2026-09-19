import { app } from "../../scripts/app.js";

function applyTextReplacements(value) {
    const utils = window.comfyAPI?.utils;
    if (!utils?.applyTextReplacements) {
        return value;
    }
    return utils.applyTextReplacements(app, value);
}

app.registerExtension({
    name: "SaveVideoFast.FilenamePrefix",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SaveVideoFast") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const widget = this.widgets?.find((w) => w.name === "filename_prefix");
            if (widget) {
                widget.serializeValue = () => applyTextReplacements(widget.value);
            }
            return result;
        };
    },
});
