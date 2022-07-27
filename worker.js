import * as Comlink from "./node_modules/comlink/dist/esm/comlink.mjs";

import init, {main, initThreadPool, dispatchChunkReceived} from "./pkg/volume_viewer.js";
import { toWrappedEvent } from "./event.js";
import { BRICK_REQUEST_EVENT, BRICK_RESPONSE_EVENT } from './js/src/volume-data-source.js';

class VolumeRenderer {
    #canvas;
    #initialized;
    #device;
    #loader;

    constructor() {
        this.#initialized = false;
        this.#canvas = null;
    }

    async initialize(offscreenCanvas) {
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
        const volumeDataSource = Comlink.wrap(worker);
        await volumeDataSource.initialize('http://localhost:8005/', 'ome-zarr/m.ome.zarr/0', null);

        this.#canvas = offscreenCanvas;
        this.#loader = {
            worker,
            volumeDataSource,
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

        this.#canvas.addEventListener('loader-request', _ => {
            (async () => {
                const chunk = await this.#loader.volumeDataSource.loadChunkAtLevel([0, 0], 2);
                dispatchChunkReceived(chunk.data, chunk.shape);

                this.#canvas.dispatchEvent(new CustomEvent(
                    BRICK_RESPONSE_EVENT,
                    {
                        detail: {
                            address: [0, 1, 2, 3],
                            brick: {
                                data: chunk.data,
                                min: 0,
                                max: 13,
                            }
                        }
                    }
                ));
            })();
        });

        this.#canvas.addEventListener(BRICK_REQUEST_EVENT, e => {
            //console.log('got brick request', e.detail, e);
            this.#loader.worker.postMessage({
                type: e.type,
                addresses: e.detail.addresses,
            });
        });
        this.#loader.worker.addEventListener('message', e => {
            console.log('message from loader', e.data.addresses[0][0]);
        })

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
