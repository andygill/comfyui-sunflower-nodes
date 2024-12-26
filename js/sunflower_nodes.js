import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "ResizeDown",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ResizeDown") {
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link, ioSlot) {
                return origOnConnectionsChange?.apply(this, arguments)
            }
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = origOnExecuted
                    ? origOnExecuted?.apply(this, arguments)
                    : undefined

                this.widgets ??= []
                this.serialize_widgets = false
                const baseOpacity = app.canvas.editor_alpha

                function processWidget(node, name, value) {

                    const current = node.widgets.find(x => x.name === name)
                    const widget = current ?? ComfyWidgets.STRING(node, name, "STRING", app).widget

                    widget.value = value
                    widget.disabled = value === null || value === undefined

                }
                processWidget(this, "output height", message.size[0])
                processWidget(this, "output width", message.size[1])

                return r
            }
        } 
    },
});
