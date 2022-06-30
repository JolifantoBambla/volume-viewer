importScripts("https://unpkg.com/comlink/dist/umd/comlink.js");
// importScripts("../../../dist/umd/comlink.js");

class Volume {
    #meta = null;

}

class DataLoader {
    #loading = [];

    constructor() {}

    async loadVolumeMetaData(path) {
        const z = await openArray({
            store,
            path: `${path}/2`,
            mode: "r"
        });

        return z.getRaw([0, 0]);
    }
}

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

    Comlink.expose(obj, port);
};

// Single line alternative:
// onconnect = (e) => Comlink.expose(obj, e.ports[0]);