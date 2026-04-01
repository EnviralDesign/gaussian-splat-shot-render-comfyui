import { app } from "../../../scripts/app.js";

const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "gaussian-splat-shot-render-comfyui";
})();

const STATE_WIDGET = "interactive_state";

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) || null;
}

function getWidgetValue(node, name, fallback = null) {
    const widget = getWidget(node, name);
    return widget ? widget.value : fallback;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) {
        return;
    }
    widget.value = value;
    widget.callback?.(value);
}

function markNodeChanged(node) {
    node.setDirtyCanvas?.(true, true);
    node.graph?.afterChange?.();
    app.graph?.afterChange?.();
    app.graph?.setDirtyCanvas?.(true, true);
}

/** Unwrap executed payloads: flat `output`, or nested `{ output: { ... } }` from some cache/history paths. */
function extractGaussianShotUi(message) {
    const out = message?.output;
    if (out && typeof out === "object" && out.ply_file) {
        return out;
    }
    if (out && typeof out === "object" && out.output && typeof out.output === "object" && out.output.ply_file) {
        return out.output;
    }
    if (message && typeof message === "object" && message.ply_file) {
        return message;
    }
    return null;
}

function firstPlyPath(ui) {
    if (!ui || typeof ui !== "object") {
        return "";
    }
    const pf = ui.ply_file;
    if (Array.isArray(pf) && pf[0]) {
        return String(pf[0]);
    }
    if (typeof pf === "string" && pf) {
        return pf;
    }
    return "";
}

/** Avoid `x || 0` so negatives and -0 round-trip from widgets correctly. */
function widgetFiniteFloat(node, name, fallback = 0) {
    const raw = getWidgetValue(node, name, fallback);
    const n = typeof raw === "number" ? raw : parseFloat(raw);
    return Number.isFinite(n) ? n : fallback;
}

function ensureWidgetDefaults(node) {
    const comboDefaults = [["background", "black", ["black", "white"]]];
    for (const [name, fallback, valid] of comboDefaults) {
        const value = getWidgetValue(node, name, fallback);
        if (!valid.includes(value)) {
            setWidgetValue(node, name, fallback);
        }
    }

    const numericDefaults = [
        ["pivot_x", 0],
        ["pivot_y", 0],
        ["pivot_z", 0],
        ["cam_yaw_deg", 0],
        ["cam_pitch_deg", 0],
        ["cam_roll_deg", 0],
        ["cam_distance", 0],
        ["output_width", 1024],
        ["output_height", 1024],
        ["gaussian_scale", 1],
        ["max_gaussians", 0],
    ];
    for (const [name, fallback] of numericDefaults) {
        const value = getWidgetValue(node, name, fallback);
        if (value === "" || Number.isNaN(Number(value))) {
            setWidgetValue(node, name, fallback);
        }
    }
}

function buildViewerState(node) {
    return {
        camera_locked: !!getWidgetValue(node, "camera_locked", false),
        pivot_x: widgetFiniteFloat(node, "pivot_x", 0),
        pivot_y: widgetFiniteFloat(node, "pivot_y", 0),
        pivot_z: widgetFiniteFloat(node, "pivot_z", 0),
        cam_yaw_deg: widgetFiniteFloat(node, "cam_yaw_deg", 0),
        cam_pitch_deg: widgetFiniteFloat(node, "cam_pitch_deg", 0),
        cam_roll_deg: widgetFiniteFloat(node, "cam_roll_deg", 0),
        cam_distance: widgetFiniteFloat(node, "cam_distance", 0),
        interactive_state: getWidgetValue(node, STATE_WIDGET, ""),
        gaussian_scale: parseFloat(getWidgetValue(node, "gaussian_scale", 1.0)) || 1.0,
        seed: parseInt(getWidgetValue(node, "seed", 0), 10) || 0,
        rand_tx_min: widgetFiniteFloat(node, "rand_tx_min", 0),
        rand_tx_max: widgetFiniteFloat(node, "rand_tx_max", 0),
        rand_ty_min: widgetFiniteFloat(node, "rand_ty_min", 0),
        rand_ty_max: widgetFiniteFloat(node, "rand_ty_max", 0),
        rand_tz_min: widgetFiniteFloat(node, "rand_tz_min", 0),
        rand_tz_max: widgetFiniteFloat(node, "rand_tz_max", 0),
        rand_yaw_min: widgetFiniteFloat(node, "rand_yaw_min", 0),
        rand_yaw_max: widgetFiniteFloat(node, "rand_yaw_max", 0),
        rand_pitch_min: widgetFiniteFloat(node, "rand_pitch_min", 0),
        rand_pitch_max: widgetFiniteFloat(node, "rand_pitch_max", 0),
        rand_roll_min: widgetFiniteFloat(node, "rand_roll_min", 0),
        rand_roll_max: widgetFiniteFloat(node, "rand_roll_max", 0),
        rand_loc_pitch_min: widgetFiniteFloat(node, "rand_loc_pitch_min", 0),
        rand_loc_pitch_max: widgetFiniteFloat(node, "rand_loc_pitch_max", 0),
        rand_loc_yaw_min: widgetFiniteFloat(node, "rand_loc_yaw_min", 0),
        rand_loc_yaw_max: widgetFiniteFloat(node, "rand_loc_yaw_max", 0),
        rand_loc_roll_min: widgetFiniteFloat(node, "rand_loc_roll_min", 0),
        rand_loc_roll_max: widgetFiniteFloat(node, "rand_loc_roll_max", 0),
        max_gaussians: (() => {
            const v = parseInt(getWidgetValue(node, "max_gaussians", 0), 10);
            return Number.isFinite(v) ? v : 0;
        })(),
        output_width: (() => {
            const v = parseInt(getWidgetValue(node, "output_width", 1024), 10);
            return Number.isFinite(v) && v > 0 ? v : 1024;
        })(),
        output_height: (() => {
            const v = parseInt(getWidgetValue(node, "output_height", 1024), 10);
            return Number.isFinite(v) && v > 0 ? v : 1024;
        })(),
        use_source_resolution: !!getWidgetValue(node, "use_source_resolution", false),
        show_viewer_hud: !!getWidgetValue(node, "show_viewer_hud", true),
    };
}

app.registerExtension({
    name: "gaussianshot.viewer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GaussianShotRender") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const plyPathWidget = getWidget(this, "ply_path");
            if (plyPathWidget) {
                plyPathWidget.hidden = true;
            }
            const stateWidget = getWidget(this, STATE_WIDGET);
            if (stateWidget) {
                stateWidget.hidden = true;
            }
            ensureWidgetDefaults(this);

            const container = document.createElement("div");
            container.style.width = "100%";
            container.style.height = "100%";
            container.style.minHeight = "420px";
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.overflow = "hidden";
            container.style.background = "#111";

            const iframe = document.createElement("iframe");
            iframe.style.width = "100%";
            iframe.style.height = "100%";
            iframe.style.border = "none";
            iframe.style.background = "#111";
            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian_shot.html?v=${Date.now()}`;
            container.appendChild(iframe);

            this.addDOMWidget("gaussian_shot_viewer", "GAUSSIAN_SHOT_VIEWER", container, {
                getValue() {
                    return "";
                },
                setValue() {},
            });

            // addDOMWidget appends; move viewer to top of widget stack (below input sockets).
            const viewerWidget = this.widgets?.find((w) => w.name === "gaussian_shot_viewer");
            if (viewerWidget && this.widgets) {
                const i = this.widgets.indexOf(viewerWidget);
                if (i > 0) {
                    this.widgets.splice(i, 1);
                    this.widgets.unshift(viewerWidget);
                }
            }
            this.setDirtyCanvas?.(true, true);

            this.gaussianShotIframe = iframe;
            this.gaussianShotUi = null;
            this.gaussianShotLoaded = false;
            this.gaussianShotLastSignature = "";
            this.gaussianShotWasLocked = !!getWidgetValue(this, "camera_locked", false);

            iframe.addEventListener("load", () => {
                this.gaussianShotLoaded = true;
                this.gaussianShotSendState(true);
            });

            this.gaussianShotSendState = (force = false) => {
                if (!this.gaussianShotLoaded || !iframe.contentWindow) {
                    return;
                }
                const state = buildViewerState(this);
                const ui = this.gaussianShotUi || {};
                const signature = JSON.stringify({ ui, state });
                if (!force && signature === this.gaussianShotLastSignature) {
                    return;
                }
                this.gaussianShotLastSignature = signature;
                iframe.contentWindow.postMessage({ type: "GAUSSIAN_SHOT_STATE", ui, state }, "*");
            };

            this.gaussianShotPoll = setInterval(() => this.gaussianShotSendState(false), 150);

            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                clearInterval(this.gaussianShotPoll);
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const ui = extractGaussianShotUi(message);
                const plyPath = firstPlyPath(ui);
                if (!ui || !plyPath) {
                    return;
                }

                const normalized = plyPath.replace(/\\/g, "/");
                const match = normalized.match(/(?:^|\/)(output|input|temp)\/(.+)$/);
                let fileUrl;
                if (match) {
                    const [, type, rest] = match;
                    const parts = rest.split("/");
                    const filename = parts.pop();
                    const subfolder = parts.join("/");
                    fileUrl = `/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                } else {
                    const filename = normalized.split("/").pop();
                    fileUrl = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                }

                this.gaussianShotUi = { ...ui, file_url: fileUrl };
                /* Do not call setSize here — same as built-in nodes: preserve user width/height after queue. */
                app.graph.setDirtyCanvas(true, true);
                this.gaussianShotSendState(true);
            };

            return result;
        };
    },

    async setup() {
        window.addEventListener("message", (event) => {
            const message = event.data;
            if (!message || message.type !== "GAUSSIAN_SHOT_CAMERA_CHANGED") {
                return;
            }

            const node = app.graph?._nodes?.find((candidate) => String(candidate.id) === String(message.node_id));
            if (!node) {
                return;
            }

            setWidgetValue(node, STATE_WIDGET, JSON.stringify(message.state));
            if (message.commit_parameters) {
                setWidgetValue(node, "pivot_x", message.state.pivot_x);
                setWidgetValue(node, "pivot_y", message.state.pivot_y);
                setWidgetValue(node, "pivot_z", message.state.pivot_z);
                setWidgetValue(node, "cam_yaw_deg", message.state.yaw_deg);
                setWidgetValue(node, "cam_pitch_deg", message.state.pitch_deg);
                setWidgetValue(node, "cam_roll_deg", message.state.roll_deg);
                setWidgetValue(node, "cam_distance", message.state.distance);
            } else if (message.set_manual_pivot) {
                setWidgetValue(node, "pivot_x", message.state.pivot_x);
                setWidgetValue(node, "pivot_y", message.state.pivot_y);
                setWidgetValue(node, "pivot_z", message.state.pivot_z);
            }
            markNodeChanged(node);
        });
    },
});
