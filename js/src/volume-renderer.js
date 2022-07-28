import * as Comlink from '../../node_modules/comlink/dist/esm/comlink.mjs';

import init, {main, initThreadPool, dispatchChunkReceived} from '../../pkg/volume_viewer.js';
import { toWrappedEvent } from './event.js';
import { BrickAddress, BRICK_REQUEST_EVENT, BRICK_RESPONSE_EVENT } from './volume-data-source.js';

export class VolumeRenderer {
    #canvas;
    #initialized;
    #device;
    #loader;

    constructor() {
        this.#initialized = false;
        this.#canvas = null;
    }

    async initialize(offscreenCanvas, config) {
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

        const worker = new Worker('./data-loading-thread.js', { type: 'module' });
        const volumeDataSource = Comlink.wrap(worker);
        const volumeMeta = await volumeDataSource.initialize(config.dataSource);

        this.#canvas = offscreenCanvas;
        this.#loader = {
            worker,
            volumeDataSource,
            volumeMeta,
        };
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

        this.#canvas.addEventListener(BRICK_REQUEST_EVENT, e => {
            (async () => {
                this.#loader.worker.postMessage({
                    type: BRICK_REQUEST_EVENT,
                    addresses: e.detail.addresses,
                });
            })()
        });
        this.#loader.worker.addEventListener('message', e => {
            if (e.data.type === BRICK_RESPONSE_EVENT) {
                console.log('message from loader', e.data);
            }
        })

        // start event loop
        main(this.#canvas, this.#loader.volumeMeta);
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
