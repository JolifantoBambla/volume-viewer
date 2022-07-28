import * as Comlink from '../../node_modules/comlink/dist/esm/comlink.mjs';

import { toWrappedEvent } from './event.js';

export class Config {
    constructor({canvas, numThreads, logging, dataSource,}) {
        this.canvas = canvas || {
            width: 800,
            height: 600,
        };
        this.numThreads = numThreads || navigator.hardwareConcurrency;
        this.logging = logging || 'error';
        this.dataSource = dataSource;
        if (!this.dataSource) {
            throw new Error("No datasource given!");
        }
    }
}

function getOrCreateCanvas(canvasConfig) {
    if (canvasConfig.canvasId) {
        return document.querySelector(canvasConfig.canvasId);
    } else {
        const parent = canvasConfig.parentId ? document.querySelector(canvasConfig.parentId) : document.body;
        const canvas = document.createElement('canvas');
        canvas.width = canvasConfig.width;
        canvas.height = canvasConfig.height;
        parent.appendChild(canvas);
        return canvas;
    }
}

export async function createOffscreenRenderer(config) {
    const canvas = getOrCreateCanvas(config.canvas);

    const hasOffscreenSupport = !!canvas.transferControlToOffscreen;
    if (hasOffscreenSupport) {
        // instantiate worker
        const worker = new Worker('./js/src/volume-renderer-thread.js', { type: "module" });
        const obj = Comlink.wrap(worker);

        // transfer control over canvas to worker
        const offscreen = canvas.transferControlToOffscreen();
        Comlink.transfer(offscreen, [offscreen]);
        await obj.initialize(offscreen, config);

        // register UI event listeners
        const dispatchToWorker = (e) => {
            e.preventDefault();
            obj.dispatchCanvasEvent(JSON.stringify(toWrappedEvent(e)));
        }
        canvas.onmousedown = dispatchToWorker;
        canvas.onmouseup = dispatchToWorker;
        canvas.onmousemove = dispatchToWorker;
        canvas.onwheel = dispatchToWorker;
        window.onkeydown = dispatchToWorker;
        window.onkeyup = dispatchToWorker;
        window.onkeypress = dispatchToWorker;

        return obj;
    } else {
        throw Error(`Canvas with id "${config.canvasId}" does not support offscreen rendering.`);
    }
}
