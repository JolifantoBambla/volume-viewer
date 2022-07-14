import * as Comlink from "./js/external/src/comlink.mjs";

import { VolumeLoader } from "./js/src/volume-data-source.js";

const volumeLoader = new VolumeLoader();

Comlink.expose(volumeLoader);
