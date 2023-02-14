import * as Comlink from '../../node_modules/comlink/dist/esm/comlink.mjs';

import init, {main, initThreadPool, dispatchChunkReceived} from '../../pkg/volume_viewer.js';
import { toWrappedEvent } from './event.js';
import { BRICK_REQUEST_EVENT, BRICK_RESPONSE_EVENT } from './volume-data-source.js';

export const RENDER_MODE_OCTREE = "octree";
export const RENDER_MODE_PAGE_TABLE = "page_table";
export const RENDER_MODES = [
    RENDER_MODE_OCTREE,
    RENDER_MODE_PAGE_TABLE
];

export class Color {
    r;
    g;
    b;

    constructor({r = 0.0, g = 0.0, b = 0.0}) {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    static random() {
        return new Color({
           r: Math.random(),
           g: Math.random(),
           b: Math.random(),
        });
    }
}

export class ChannelSettings {
    channelName;
    channelIndex;
    maxLoD;
    minLoD;
    lodFactor;
    thresholdLower;
    thresholdUpper;
    color;
    visible;

    constructor({channelName, channelIndex = 0, maxLoD = 0, minLoD, lodFactor = 1.0, thresholdLower = 0.01, thresholdUpper = 1.0, color, visible = false}) {
        // todo: validation
        this.channelName = channelName || `${channelIndex}`;
        this.channelIndex = channelIndex;
        this.maxLoD = maxLoD;
        this.minLoD = minLoD || maxLoD;
        this.lodFactor = lodFactor;
        this.thresholdLower = thresholdLower;
        this.thresholdUpper = thresholdUpper;
        this.color = color || Color.random();
        this.visible = visible;
    }
}

export class VolumeRendererCreateOptions {
    maxVisibleChannels;
    maxResolutions;

    constructor({maxVisibleChannels = 17, maxResolutions = 15}) {
        this.maxVisibleChannels = maxVisibleChannels;
        this.maxResolutions = maxResolutions;
    }
}

export class VolumeRendererSettings {
    createOptions;
    renderMode;
    stepScale;
    maxSteps;
    backgroundColor;
    channelSettings;

    constructor({createOptions = new VolumeRendererCreateOptions({}), renderMode = RENDER_MODE_OCTREE, stepScale = 1.0, maxSteps = 300, backgroundColor = new Color({}),channelSettings = [new ChannelSettings({})]}) {
        // todo: validation
        this.createOptions = createOptions;
        this.renderMode = renderMode;
        this.stepScale = stepScale;
        this.maxSteps = maxSteps;
        this.backgroundColor = backgroundColor;
        this.channelSettings = channelSettings;
    }
}

export class VolumeRenderer {
    #canvas;
    #initialized;
    #loader;

    constructor() {
        this.#initialized = false;
        this.#canvas = null;
    }

    async initialize(offscreenCanvas, config) {
        // to accept an OffscreenCanvas as a raw window handle, winit needs some properties to exist on both the window and
        // the canvas. This is a hack to make sure that the window has the properties that winit needs.
        // see: https://github.com/rust-windowing/winit/issues/1518
        self.Window = WorkerGlobalScope;
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

    async run(renderSettings = null) {
        if (!this.#initialized) {
            console.error('\'run\' called on uninitialized VolumeRenderer.')
            return;
        }

        // initialize wasm (including module specific initialization)
        await init();

        // initialize the thread pool using all available cores
        // todo: make configurable
        await initThreadPool(navigator.hardwareConcurrency);

        this.#canvas.addEventListener(BRICK_REQUEST_EVENT, e => {
            (async () => {
                this.#loader.worker.postMessage({
                    type: BRICK_REQUEST_EVENT,
                    addresses: e.detail.get('addresses'),
                });
            })();
        });
        this.#loader.worker.addEventListener('message', e => {
            if (e.data.type === BRICK_RESPONSE_EVENT) {
                // todo: processing a custom event is really slow, try other approaches:
                //  - write to texture in JS and pass texture handle to canvas (copyTextureToTexture on wasm side then)
                //  - call into wasm directly by calling a global event loop proxy or a global reference to the data source
                this.#canvas.dispatchEvent(
                    new CustomEvent(
                        BRICK_RESPONSE_EVENT,
                        {
                            detail: {
                                address: e.data.brick.address,
                                brick: e.data.brick.brick,
                            }
                        }
                    )
                );
            }
        });

        // start event loop
        main(this.#canvas, this.#loader.volumeMeta, renderSettings);
    }

    get volumeMeta() {
        return this.#loader.volumeMeta;
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

    dispatchUIEvent(eventString) {
        if (this.#canvas) {
            const event = JSON.parse(eventString);
            const detail = {};
            detail[`${event.setting}`] = event.value;
            this.#canvas.dispatchEvent(new CustomEvent(event.type, { detail }));
        }
    }
}
