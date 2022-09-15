import init, {initThreadPool, convertToUint8, maxValue, isEmpty} from "../../pkg/volume_viewer.js";

import { openGroup, openArray, slice } from '../../node_modules/zarr/zarr.mjs';

// todo: maybe typescript?

export const BRICK_REQUEST_EVENT = 'data-loader:brick-request';
export const BRICK_RESPONSE_EVENT = 'data-loader:brick-response';

export class BrickAddress {
    constructor(index, level, channel = 0) {
        this.index = index;
        this.level = level;
        this.channel = channel;
    }
}

export class VolumeResolutionMeta {
    constructor(brickSize, volumeSize, scale) {
        this.volumeSize = volumeSize;
        this.paddedVolumeSize = [];
        for (let i = 0; i < 3; ++i) {
            this.paddedVolumeSize.push(
                Math.ceil(volumeSize[i] / brickSize[i]) * brickSize[i]
            );
        }
        this.scale = scale;
        console.log("scale:", scale);
    }
}

export class MultiResolutionVolumeMeta {
    constructor(brickSize, scale, resolutions) {
        this.brickSize = brickSize;
        this.scale = scale;
        this.resolutions = resolutions;
    }
}

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

/**
 * An interface for brick loading.
 */
export class BrickLoader {
    /**
     * Loads a chunk of the volume at a given resolution level
     * @param brickIndex
     * @param level
     * @param channel
     * @returns {Promise<RawVolumeChunk>}
     */
    async loadBrick(brickAddress) {
        throw new Error("not implemented");
    }

    /**
     * Returns the size of a single brick.
     */
    get brickSize() {
        throw new Error("not implemented");
    }
}

/**
 * An interface for volume date sources
 */
export class VolumeDataSource extends BrickLoader {
    #config;

    constructor(config) {
        super();
        this.#config = config;
    }

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

    get volumeMeta() {
        throw new Error("not implemented");
    }

    async getMaxValue() {
        throw new Error("not implemented");
    }

    get config() {
        return this.#config;
    }
}

export class OmeZarrDataSource extends VolumeDataSource {
    #zarrGroup;
    #omeZarrMeta;
    #zarrArrays;

    #volumeMeta;
    #maxValues;

    #emptyBricks;

    constructor(zarrGroup, omeZarrMeta, zarrArrays, config) {
        super(config);

        this.#zarrGroup = zarrGroup;
        this.#omeZarrMeta = omeZarrMeta;
        this.#zarrArrays = zarrArrays;

        const brickSize = config.minimumBrickSize;
        for (const arr of this.#zarrArrays) {
            for (let i = 0; i < 3; ++i) {
                brickSize[i] = Math.min(
                    config.maximumBrickSize[i],
                    arr.shape[arr.shape.length - 1 - i],
                    Math.max(
                        config.minimumBrickSize[i],
                        brickSize[i],
                        arr.chunks[arr.chunks.length - 1 - i]
                    )
                );
            }
        }

        const resolutions = [];
        for (let i = 0; i < this.#zarrArrays.length; ++i) {
            const volumeSize = this.#zarrArrays[i]
                .shape
                .slice()
                .reverse()
                .slice(0, 3);
            // todo: this is probably not exactly what I want
            //  I need this to be the scale factor by which I can scale the [0,1]^3 box
            //  this is independent of the volumeSize, which is in voxels
            const scale = this.#omeZarrMeta
                .multiscales[0]
                .datasets[i]
                .coordinateTransformations
                .find(d => d.type === 'scale')
                .scale
                .slice()
                .reverse()
                .slice(0, 3);
            resolutions.push(
                new VolumeResolutionMeta(brickSize, volumeSize, scale)
            );
        }
        console.log(this.#omeZarrMeta);
        const scale = this.#omeZarrMeta
            .multiscales[0]
            .coordinateTransformations
            .find(d => d.type === 'scale')
            .scale
            .slice()
            .reverse()
            .slice(0, 3);

        this.#volumeMeta = new MultiResolutionVolumeMeta(brickSize, scale, resolutions);
        this.#maxValues = new Array(this.#zarrArrays[0].shape.slice().reverse()[3]);
        this.#emptyBricks = {};
    }

    get brickSize() {
        return this.#volumeMeta.brickSize;
    }

    get volumeMeta() {
        return this.#volumeMeta;
    }

    #ensureSafeEmptyBrickAccess(brickAddress) {
        const c = `${brickAddress.channel}`;
        const l = `${brickAddress.level}`;
        if (this.#emptyBricks[c] === undefined) {
            this.#emptyBricks[c] = {}
        }
        if (this.#emptyBricks[c][l] === undefined) {
            this.#emptyBricks[c][l] = {}
        }
    }

    #setBrickEmpty(brickAddress) {
        this.#ensureSafeEmptyBrickAccess(brickAddress);
        const c = `${brickAddress.channel}`;
        const l = `${brickAddress.level}`;
        const i = `${brickAddress.index}`;
        this.#emptyBricks[c][l][i] = true;
    }

    isBrickEmpty(brickAddress) {
        this.#ensureSafeEmptyBrickAccess(brickAddress);
        const c = `${brickAddress.channel}`;
        const l = `${brickAddress.level}`;
        const i = `${brickAddress.index}`;
        return this.#emptyBricks[c][l][i] === true;
    }

    async loadBrick(brickAddress) {
        if (this.isBrickEmpty(brickAddress)) {
            return null;
        }

        const brickSelection = [];
        for (let i = 2; i >= 0; --i) {
            const origin = brickAddress.index[i] * this.brickSize[i];
            brickSelection.push(
                slice(origin, origin + this.brickSize[i])
            );
        }
        const raw = await this.#zarrArrays[brickAddress.level]
            .getRaw([0, brickAddress.channel, ...brickSelection]);

        if (isEmpty(raw.data)) {
            this.#setBrickEmpty(brickAddress);
            return null;
        }

        // this uses multithreading to transform data from uint16 to uint8
        const uint8 = convertToUint8(
            raw.data,
            await this.getMaxValue(brickAddress.channel)
        );

        return {
            ...raw,
            data: uint8,
        };
    }

    async getMaxValue(channel = 0) {
        if (!this.#maxValues[channel]) {
            this.#maxValues[channel] = maxValue(
                (await this.#zarrArrays
                        .slice()
                        .reverse()[0]
                        .getRaw([0, channel])
                ).data
            );
        }
        return this.#maxValues[channel];
    }
}

export class VolumeDataSourceConfig {
    store;
    path;
    sourceType;
    maximumBrickSize;
    minimumBrickSize;

    constructor({store, path, sourceType = "OME-Zarr"}, { maximumBrickSize = [256, 256, 256], minimumBrickSize = [32, 32, 32] }) {
        this.store = store;
        this.path = path;
        this.sourceType = sourceType;
        this.maximumBrickSize = maximumBrickSize;
        this.minimumBrickSize = minimumBrickSize;
    }
}

async function createVolumeDataSource(config) {
    const mode = 'r';

    const {store, path} = config;
    const group = await openGroup(store, path, mode);
    const attributes = await group.attrs.asObject();
    const multiscale = attributes.multiscales[0];
    const resolutions = [];
    for (const dataset of multiscale.datasets) {
        resolutions.push(await openArray({
            store: config.store,
            path: `${path}/${dataset.path}`,
            mode
        }));

        const idx = resolutions.length - 1;
        console.log(`res shape ${resolutions[idx].shape}, res chunks: ${resolutions[idx].chunks}, res chunkSize: ${resolutions[idx].chunkSize}, res numChunks: ${resolutions[idx].numChunks}`)
    }

    return new OmeZarrDataSource(group, attributes, resolutions, config);
}

export class VolumeLoader {
    #initialized;
    #dataSource;
    #postMessage;

    #currentlyLoading;

    constructor(postMessage) {
        this.#initialized = false;
        this.#dataSource = null;
        this.#postMessage = postMessage;
        this.#currentlyLoading = {};
    }

    async initialize(dataSourceConfig) {
        if (this.#initialized) {
            throw Error("VolumeLoader is already initialized. Call reset instead!");
        }

        await init();
        await initThreadPool(navigator.hardwareConcurrency);

        this.#dataSource = await createVolumeDataSource(dataSourceConfig);

        this.#currentlyLoading = {};
        this.#initialized = true;

        return this.#dataSource.volumeMeta;
    }

    async reset(store, path, dataSourceType = null) {
        this.#initialized = false;
        await this.initialize(store, path, dataSourceType);
    }

    get initialized() {
        return this.#initialized;
    }

    isLoadingChunk(brickAddress) {
        return this.#currentlyLoading[JSON.stringify(brickAddress)] === true;
    }

    /**
     * Loads a chunk of the volume at a given resolution level
     * @param brickAddress a `BrickAddress`
     */
    async loadBrick(brickAddress) {
        if (!this.#initialized) {
            throw Error("Can't load chunk from uninitialized data source");
        }
        if (this.isLoadingChunk()) {
            return;
        }
        this.#currentlyLoading[JSON.stringify(brickAddress)] = true;
        setTimeout(() => {
            (async () => {
                this.#postMessage({
                    type: BRICK_RESPONSE_EVENT,
                    brick: {
                        address: brickAddress,
                        data: await this.#dataSource.loadBrick(brickAddress) || [],
                    }
                });
                this.#currentlyLoading[JSON.stringify(brickAddress)] = false;
            })()
        }, Math.random() * 10);
    }

    async handleExternEvent(e) {
        if (e.type === BRICK_REQUEST_EVENT) {
            Promise.all(e.addresses.map(async a => this.loadBrick(a)))
                .catch(console.err);
        }
    }
}
