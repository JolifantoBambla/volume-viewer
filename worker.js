import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";
// importScripts("../../../dist/umd/comlink.js");

import init, {testDeviceSharing, main} from "./pkg/volume_viewer.js";

import { openArray } from 'https://cdn.skypack.dev/zarr';

import { toWrappedEvent } from "./event.js";

async function getRawZarrArray() {
    const store = 'http://localhost:8005/';
    const path = 'ome-zarr/m.ome.zarr/0';

    const z = await openArray({
        store,
        path: `${path}/2`,
        mode: "r"
    });

    return await z.getRaw([0, 0]);
}

const obj = {
    counter: 0,
    canvas: null,
    inc() {
        this.counter++;
    },
    initialize(offscreenCanvas) {
        // This is a hack so that wgpu can create an instance from a dedicated worker
        // See: https://github.com/gfx-rs/wgpu/issues/1986
        self.Window = WorkerGlobalScope;

        // to accept an OffscreenCanvas as a raw window handle, winit needs some properties to exist on both the window and
        // the canvas. This is a hack to make sure that the window has the properties that winit needs.
        // see: https://github.com/rust-windowing/winit/issues/1518
        self.Window.prototype.devicePixelRatio = 1;
        offscreenCanvas.setAttribute = (name, value) => {};
        offscreenCanvas.style = {
            setProperty(name, value) {},
        };

        this.canvas = offscreenCanvas;
    },
    async run() {
        // initialize wasm (including module specific initialization)
        await init();

        // this is just a test, can be removed
        testDeviceSharing();

        const {data, shape} = await getRawZarrArray();
        main(data, shape, this.canvas);
    },
    dispatchCanvasEvent(eventString) {
        if (this.canvas) {
            this.canvas.dispatchEvent(toWrappedEvent(JSON.parse(eventString)));
        }
    }
};

Comlink.expose(obj);

/**
 * When a connection is made into this shared worker, expose `obj`
 * via the connection `port`.
 */
//onconnect = function (event) {
//    const port = event.ports[0];
//    Comlink.expose(obj, port);
//};

// Single line alternative:
// onconnect = (e) => Comlink.expose(obj, e.ports[0]);