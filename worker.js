import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";
// importScripts("../../../dist/umd/comlink.js");

import init, {testDeviceSharing, runOffscreenExample} from "./pkg/volume_viewer.js";

import { openArray } from 'https://cdn.skypack.dev/zarr';

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

async function runRenderer(offscreenCanvas) {
    // This is a hack so that wgpu can create an instance from a dedicated worker
    // See: https://github.com/gfx-rs/wgpu/issues/1986
    //self.Window = WorkerGlobalScope;
    console.log('how you doin');

    console.log('init');
    // initialize wasm (including module specific initialization)
    await init();

    console.log('share device');

    testDeviceSharing();

    console.log("running offscreen example");
    console.log(offscreenCanvas);

    const {data, shape} = await getRawZarrArray();
    await runOffscreenExample(data, shape, offscreenCanvas);
}

const obj = {
    counter: 0,
    inc() {
        this.counter++;
    },
    run(offscreen, adapter) {
        console.log("running", navigator.gpu, 'asdf', adapter);
        runRenderer(offscreen);
    },
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