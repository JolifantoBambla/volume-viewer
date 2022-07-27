import * as Comlink from "./js/external/src/comlink.mjs";

import { VolumeLoader } from "./js/src/volume-data-source.js";

const volumeLoader = new VolumeLoader((message, transfer) => self.postMessage(message, transfer));

Comlink.expose(volumeLoader);

self.onmessage = function handleEvent({data: event}) {
    volumeLoader.handleExternEvent(event)
        .catch(console.error);
};
