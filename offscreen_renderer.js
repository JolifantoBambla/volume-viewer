import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";
import { toWrappedEvent } from "./event.js";

// todo: define config options
//  - canvas
//    - parentId: string | null (default: document.body)
//    - canvasId: string | null (default: create new canvas & attach to config.canvas.parentId)
//    - width: uint
//    - height: uint
//  - numThreads: uint | null (default: navigator.hardwareConcurrency)
//  - logLevel: string | null (default: error)


export async function createOffscreenRenderer(config) {
    const canvas = document.querySelector(config.canvasId);

    const hasOffscreenSupport = !!canvas.transferControlToOffscreen;
    if (hasOffscreenSupport) {
        // instantiate worker
        const worker = new Worker("worker.js", { type: "module" });
        const obj = Comlink.wrap(worker);

        // transfer control over canvas to worker
        const offscreen = canvas.transferControlToOffscreen();
        Comlink.transfer(offscreen, [offscreen]);
        obj.initialize(offscreen);

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
