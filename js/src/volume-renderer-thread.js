import * as Comlink from '../../node_modules/comlink/dist/esm/comlink.mjs';

import { VolumeRenderer } from './volume-renderer.js';

const renderer = new VolumeRenderer((message, transfer) => self.postMessage(message, transfer));

Comlink.expose(renderer);
