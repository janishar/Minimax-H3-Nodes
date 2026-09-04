import { app } from "../../scripts/app.js";

// Grows H3CombineLatents' input sockets as they're wired up: connecting the
// last "latent_N" socket adds a fresh empty one after it, and disconnecting
// prunes back down to exactly one empty trailing socket. latent_1/latent_2
// are the node's required Python inputs and are never removed.

const NODE_NAME = "H3CombineLatents";
const PREFIX = "latent_";
const TYPE = "LATENT";
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
    name: "MinimaxH3.CombineLatentsDynamicInputs",
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
