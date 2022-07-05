import * as Comlink from "https://unpkg.com/comlink/dist/esm/comlink.mjs";

const obj = {
    counter: 0,
    inc() {
        this.counter++;
    },
};

/**
 * When a connection is made into this shared worker, expose `obj`
 * via the connection `port`.
 */
onconnect = function (event) {
    const port = event.ports[0];

    console.log('connected');

    Comlink.expose(obj, port);
};

// Single line alternative:
// onconnect = (e) => Comlink.expose(obj, e.ports[0]);