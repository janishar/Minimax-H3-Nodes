import { app } from "../../scripts/app.js";

// Grows H3GateSwitch's input sockets as they're wired up: connecting the
// last "value_N" socket adds a fresh empty one after it, and disconnecting
// prunes back down to MIN_INPUTS empty trailing sockets. Same mechanism as
// combine_latents.js -- see that file's module docstring. MIN_INPUTS/
// MAX_INPUTS here must match gates/gate_switch.py's constants of the same
// name.

const NODE_NAME = "H3GateSwitch";
const PREFIX = "value_";
const TYPE = "*";
const MIN_INPUTS = 2;
const MAX_INPUTS = 32;

function stabilize(node) {
    if (!node.inputs) return;

    for (let i = node.inputs.length - 1; i >= MIN_INPUTS; i--) {
        if (node.inputs[i].link == null) node.removeInput(i);
    }

    if (node.inputs.length < MAX_INPUTS) {
        const last = node.inputs[node.inputs.length - 1];
        if (!last || last.link != null) {
            node.addInput(`${PREFIX}${node.inputs.length + 1}`, TYPE);
        }
    }

    // Renumber for a clean sequential display -- safe, since litegraph links
    // bind to slot index, not input name.
    node.inputs.forEach((input, i) => {
        input.name = `${PREFIX}${i + 1}`;
    });

    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MinimaxH3.GateSwitchDynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            stabilize(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            const result = onConnectionsChange?.apply(this, arguments);
            if (type === LiteGraph.INPUT) {
                // Deferred: mutating node.inputs synchronously inside this
                // callback can race litegraph's own connection bookkeeping.
                setTimeout(() => stabilize(this), 0);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            stabilize(this);
            return result;
        };
    },
});
