import init, { initThreadPool, convertToUint8, maxValue } from "../../pkg/volume_viewer.js";

import { openGroup, openArray } from '../../node_modules/zarr/zarr.mjs';

// todo: maybe typescript?

export const BRICK_REQUEST_EVENT = 'data-loader:brick-request';
export const BRICK_RESPONSE_EVENT = 'data-loader:brick-response';

// Note: bioformats2raw only generates 2D chunks
// we use the following brick size for our dataset:
//   (
//      resolutions[0].chunkSize.x,
//      resolutions[0].chunkSize.y,
//      min(resolutions[0].shape.z,
//          min(resolutions[0].chunkSize.x,
//              resolutions[0].chunkSize.y
//          )
//      )
//   )


export class Chunk {
    /**
     * The canonical origin of the chunk (i.e. in [0,1]^3)
     */
    origin;

    /**
     * The extent of the chunk (within [0,1]^3)
     */
    extent;
}

export class RawVolumeChunk {
    data;
    shape;
    chunk;
}

export class ChunkLoader {
    /**
     * Loads a chunk of the volume at a given resolution level
     * @param level
     * @param chunk
     * @returns {Promise<RawVolumeChunk>}
     */
    async loadChunkAtLevel(chunk, level) {
        throw new Error("not implemented");
    }
}

export class VolumeDataSource extends ChunkLoader {
    /**
     * Translates a canonical address within the volume (i.e. within [0,1]^3) to a physical address within the array.
     * @param address the virtual address to translate
     * @param level the resolution level
     * @returns {Array<number>}
     */
    translateVirtualAddress(address, level) {
        const scale = this.scaleAtResolution(level)
        const physicalAddress = [];
        for (let i = 0; i < scale.length; ++i) {
            physicalAddress.push(address[i] * scale[i]);
        }
        return physicalAddress;
    }

    /**
     * Returns the scale of the volume at the given resolution level.
     * @param level the resolution level
     * @returns {Array<number>}
     */
    scaleAtResolution(level) {
        throw new Error("not implemented");
    }
}

export class OmeZarrDataSource extends VolumeDataSource {
    #zarrGroup;
    #omeZarrMeta;
    #zarrArrays;

    constructor(zarrGroup, omeZarrMeta, zarrArrays) {
        super();
        this.#zarrGroup = zarrGroup;
        this.#omeZarrMeta = omeZarrMeta;
        this.#zarrArrays = zarrArrays;
    }

    async loadChunkAtLevel(chunk, level) {
        // todo: translate volume chunk address
        const raw = await this.#zarrArrays[level].getRaw(chunk);
        // todo: cast to u8 and normalize

        // this test uses multithreading to transform data from uint16 to uint8
        const uint8 = convertToUint8(raw.data, maxValue(raw.data));

        return {
            ...raw,
            data: uint8,
        };
    }
}

async function createVolumeDataSource(store, path, dataSourceType) {
    const mode = 'r';

    const group = await openGroup(store, path, mode);
    const attributes = await group.attrs.asObject();
    const multiscale = attributes.multiscales[0];
    const resolutions = [];
    for (const dataset of multiscale.datasets) {
        resolutions.push(await openArray({
            store,
            path: `${path}/${dataset.path}`,
            mode
        }));

        const idx = resolutions.length - 1;
        console.log(`res shape ${resolutions[idx].shape}, res chunks: ${resolutions[idx].chunks}, res chunkSize: ${resolutions[idx].chunkSize}, res numChunks: ${resolutions[idx].numChunks}`)
    }
    return new OmeZarrDataSource(group, attributes, resolutions);
}

export class VolumeLoader {
    #initialized;
    #dataSource;
    #postMessage;

    constructor(postMessage) {
        this.#initialized = false;
        this.#dataSource = null;
        this.#postMessage = postMessage;
    }

    async initialize(store, path, dataSourceType = null) {
        if (this.#initialized) {
            throw Error("VolumeLoader is already initialized. Call reset instead!");
        }

        await init();
        await initThreadPool(navigator.hardwareConcurrency);

        this.#dataSource = await createVolumeDataSource(store, path, dataSourceType);
        this.#initialized = true;
    }

    async reset(store, path, dataSourceType = null) {
        this.#initialized = false;
        await this.initialize(store, path, dataSourceType);
    }

    get initialized() {
        return this.#initialized;
    }

    isLoadingChunk(chunk, level) {
        return false;
    }

    /**
     * Loads a chunk of the volume at a given resolution level
     * @param chunk
     * @param level
     */
    async loadChunkAtLevel(chunk, level) {
        if (!this.#initialized) {
            throw Error("Can't load chunk from uninitialized data source");
        }
        if (this.isLoadingChunk(chunk, level)) {
            return;
        }
        // todo: this is very naive - should probably be in a task queue and keep track / cache volume chunks
        return this.#dataSource.loadChunkAtLevel(chunk, level);
    }

    async handleExternEvent(e) {
        if (e.type === BRICK_REQUEST_EVENT) {
            setTimeout(() => {
                console.log('handling external event', e.addresses[0][0]);
                this.#postMessage(e);
            }, Math.random() * 10000); // this is a test to make sure the message handling is actually async -> it is!
        }

    }
}
