import * as Comlink from '../../node_modules/comlink/dist/esm/comlink.mjs';

import { VolumeLoader } from './volume-data-source.js';

const volumeLoader = new VolumeLoader((message, transfer) => self.postMessage(message, transfer));

Comlink.expose(volumeLoader);

self.onmessage = function handleEvent({data: event}) {
    volumeLoader.handleExternEvent(event)
        .catch(console.error);
};
