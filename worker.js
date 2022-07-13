import * as Comlink from "./external-js/src/comlink.mjs";

import init, {testDeviceSharing, main, initThreadPool} from "./pkg/volume_viewer.js";
import { toWrappedEvent } from "./event.js";

class VolumeRenderer {
    #canvas;
    #initialized;
    #device;
    #loader;

    constructor() {
        this.#initialized = false;
        this.#canvas = null;
    }

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

        const worker = new Worker('./loader.js', { type: 'module' });

        this.#canvas = offscreenCanvas;
        this.#loader = worker;
        this.#initialized = true;
    }

    async run() {
        if (!this.#initialized) {
            console.warn('\'run\' called on uninitialized VolumeRenderer.')
            return;
        }

        // initialize wasm (including module specific initialization)
        await init();

        // initialize the thread pool using all available cores
        // todo: make configurable
        await initThreadPool(navigator.hardwareConcurrency);

        // note this is used to share a GPUDevice handle with the JS side
        this.#canvas.addEventListener('from-rust', e => {
            this.#device = e.detail;
            console.log('device limits', this.#device.limits);
        });

        this.#canvas.addEventListener('loader-request', e => {
            this.#loader.postMessage(e.detail);
        });
        this.#loader.addEventListener('message', e => {
            console.log('got event', e);
            this.#canvas.dispatchEvent(new CustomEvent('zarr:group', { detail: e.detail }));
        })

        // this is just a test, can be removed
        testDeviceSharing();

        // start event loop
        main(this.#canvas);
    }

    /**
     * Dispatches a serialized event to the OffscreenCanvas.
     * @param eventString a serialized event
     */
    dispatchCanvasEvent(eventString) {
        if (this.#canvas) {
            this.#canvas.dispatchEvent(toWrappedEvent(JSON.parse(eventString)));
        }
    }
}

const renderer = new VolumeRenderer();

Comlink.expose(renderer);
