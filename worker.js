import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";
// importScripts("../../../dist/umd/comlink.js");

import init, {testDeviceSharing, runOffscreenExample} from "./pkg/volume_viewer.js";

import { openArray } from 'https://cdn.skypack.dev/zarr';

/*
interface MouseEvent extends Event {
    altKey: boolean;
    button: number;
    buttons: number;
    clientX: number;
    clientY: number;
    ctrlKey: boolean;
    layerX: number;
    layerY: number;
    metaKey: boolean;
    movementX: number;
    movementY: number;
    offsetX: number;
    offsetY: number;
    pageX: number;
    pageY: number;
    relatedTarget: EventTarget;
    screenX: number;
    screenY: number;
    mozInputSource: number;
    webkitForce: number;
    x: number;
    y: number;
}
*/

class MouseEvent extends Event {
    altKey;
    button;
    buttons;
    clientX;
    clientY;
    ctrlKey;
    layerX;
    layerY;
    metaKey;
    movementX;
    movementY;
    offsetX;
    offsetY;
    pageX;
    pageY;
    relatedTarget;
    screenX;
    screenY;
    mozInputSource;
    webkitForce;
    x;
    y;

    constructor(type, options) {
        super(type, options);
        this.altKey = options.altKey;
        this.button = options.button;
        this.buttons = options.buttons;
        this.clientX = options.clientX;
        this.clientY = options.clientY;
        this.ctrlKey = options.ctrlKey;
        this.layerX = options.layerX;
        this.layerY = options.layerY;
        this.metaKey = options.metaKey;
        this.movementX = options.movementX;
        this.movementY = options.movementY;
        this.offsetX = options.offsetX;
        this.offsetY = options.offsetY;
        this.pageX = options.pageX;
        this.pageY = options.pageY;
        this.relatedTarget = options.relatedTarget;
        this.screenX = options.screenX;
        this.screenY = options.screenY;
        this.mozInputSource = options.mozInputSource;
        this.webkitForce = options.webkitForce;
        this.x = options.x;
        this.y = options.y;
    }
}

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
    self.Window = WorkerGlobalScope;

    // to accept an OffscreenCanvas as a raw window handle, winit needs some properties to exist on both the window and
    // the canvas. This is a hack to make sure that the window has the properties that winit needs.
    // see: https://github.com/rust-windowing/winit/issues/1518
    self.Window.prototype.devicePixelRatio = 1;
    offscreenCanvas.setAttribute = (name, value) => {};
    offscreenCanvas.style = {
        setProperty(name, value) {},
    };

    // initialize wasm (including module specific initialization)
    await init();

    // this is just a test, can be removed
    testDeviceSharing();

    const {data, shape} = await getRawZarrArray();
    await runOffscreenExample(data, shape, offscreenCanvas);
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
        this.canvas.addEventListener('mousedown', (e) => {
            console.log('custom listener', e);
        });
        this.canvas.onmousedown = (e) => {
            console.log('on mouse down!', e);
        }
    },
    async run() {
        // initialize wasm (including module specific initialization)
        await init();

        // this is just a test, can be removed
        testDeviceSharing();

        const {data, shape} = await getRawZarrArray();
        runOffscreenExample(data, shape, this.canvas);
    },
    dispatchCanvasEvent(eventString) {
        const eventFromMain = JSON.parse(eventString);
        console.log('event in worker', eventFromMain);
        if (this.canvas) {
            const event = new Event(eventFromMain.type, eventFromMain);
            const customMouseEvent = new MouseEvent(eventFromMain.type, eventFromMain);
            //customMouseEvent.prototype.constructor = 'MouseEvent';
            console.log(event);
            console.log(customMouseEvent);

            //eventFromMain.prototype.constructor.name = 'MouseEvent';
            this.canvas.dispatchEvent(event);
            this.canvas.dispatchEvent(customMouseEvent);
            //this.canvas.dispatchEvent(eventFromMain);
            console.log(self);
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