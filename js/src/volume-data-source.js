import init, {
    initThreadPool,
    isZeroU8, isZeroU16, isZeroF32,
    maxU8, maxU16, maxF32,
    scaleU8ToU8, scaleU16ToU8, scaleF32ToU8,
    logTransformU8, logTransformU16, logTransformF32,
    logScaleU8, logScaleU16, logScaleF32
} from "../../pkg/volume_viewer.js";

import { openGroup, openArray, slice } from '../../node_modules/zarr/zarr.mjs';

// todo: maybe typescript?

/**
 * Splits a numpy data type descriptor (typestr) into its byte order and data type portion.
 * @param dtype the numpy data type descriptor to split.
 * @returns {{type: string, byteOrder: string}}
 */
const splitDtype = dtype => {
    return {
        byteOrder: dtype.slice(0, 1),
        type: dtype.slice(1),
    }
};

const getMaxValueForDataTypeDescriptor = dtypeDescriptor => {
    switch (dtypeDescriptor.type) {
        case 'u1':
            return 255.;
        case 'u2':
            return 65535.;
        case 'f4':
            return Number.MAX_VALUE;
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${dtypeDescriptor.type}`);
    }
};

const isZero = (data, dtypeDescriptor, threshold) => {
    switch (dtypeDescriptor.type) {
        case 'u1':
            return isZeroU8(data, threshold);
        case 'u2':
            return isZeroU16(data, threshold);
        case 'f4':
            return isZeroF32(data, threshold);
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${dtypeDescriptor.type}`);
    }
};

const castToU8 = (data, dtypeDescriptor) => {
    switch (dtypeDescriptor.type) {
        case 'u1':
            return data;
        case 'u2':
            return scaleU16ToU8(data, getMaxValueForDataTypeDescriptor(dtypeDescriptor));
        case 'f4':
            return scaleF32ToU8(data, getMaxValueForDataTypeDescriptor(dtypeDescriptor));
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${dtypeDescriptor.type}`);
    }
};

const logTransformToU8 = (data, dtypeDescriptor) => {
    const maxValue = getMaxValueForDataTypeDescriptor(dtypeDescriptor);
    switch (dtypeDescriptor.type) {
        case 'u1':
            return logScaleU8(data, maxValue);
        case 'u2':
            return logScaleU16(data, maxValue);
        case 'f4':
            return logScaleF32(data, maxValue);
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${dtypeDescriptor.type}`);
    }
};

export const BRICK_REQUEST_EVENT = 'data-loader:brick-request';
export const BRICK_RESPONSE_EVENT = 'data-loader:brick-response';

export const PREPROCESS_METHOD_CAST = 'cast';
export const PREPROCESS_METHOD_SCALE_TO_MAX = 'scaleToMax';
export const PREPROCESS_METHOD_LOG = 'log';
export const PREPROCESS_METHOD_LOG_SCALE = 'logScale';

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

    get isZeroThreshold() {
        return this.#config.isZeroThreshold || 0.0;
    }

    get preprocessingMethod() {
        return this.#config.preprocessingMethod || PREPROCESS_METHOD_CAST;
    }

    async #scaleToMax(data, dtypeDescriptor) {
        const maxValue = await this.getMaxValue();
        switch (dtypeDescriptor.type) {
            case 'u1':
                return scaleU8ToU8(data, maxValue);
            case 'u2':
                return scaleU16ToU8(data, maxValue);
            case 'f4':
                return scaleF32ToU8(data, maxValue);
            default:
                throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${dtypeDescriptor.type}`);
        }
    }

    async #logScale(data, dtypeDescriptor) {
        const maxValue = await this.getMaxValue();
        switch (dtypeDescriptor.type) {
            case 'u1':
                return logScaleU8(data, maxValue);
            case 'u2':
                return logScaleU16(data, maxValue);
            case 'f4':
                return logScaleF32(data, maxValue);
            default:
                throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${dtypeDescriptor.type}`);
        }
    }

    async preprocess(data, dtypeDescriptor) {
        switch (this.preprocessingMethod) {
            case PREPROCESS_METHOD_LOG_SCALE:
                return this.#logScale(data, dtypeDescriptor);
            case PREPROCESS_METHOD_SCALE_TO_MAX:
                return this.#scaleToMax(data, dtypeDescriptor);
            case PREPROCESS_METHOD_LOG:
                return logTransformToU8(data, dtypeDescriptor);
            case PREPROCESS_METHOD_CAST:
            default:
                return castToU8(data, dtypeDescriptor);
        }
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

        console.log(omeZarrMeta);
        console.log(zarrArrays);

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
        const globalCoordinateTransformations = this.#omeZarrMeta
            .multiscales[0]
            .coordinateTransformations;
        let scale = [1., 1., 1.];
        if (globalCoordinateTransformations) {
            scale = globalCoordinateTransformations.find(d => d.type === 'scale')
                .scale
                .slice()
                .reverse()
                .slice(0, 3);
        }

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
        if (!this.isBrickEmpty(brickAddress)) {
            const brickSelection = [];
            for (let i = 2; i >= 0; --i) {
                const origin = brickAddress.index[i] * this.brickSize[i];
                brickSelection.push(
                    slice(origin, origin + this.brickSize[i])
                );
            }
            const dtypeDescriptor = splitDtype(this.#zarrArrays[brickAddress.level].meta.dtype);

            const raw = await this.#zarrArrays[brickAddress.level]
                .getRaw([0, brickAddress.channel, ...brickSelection]);

            if (isZero(raw.data, dtypeDescriptor, this.isZeroThreshold)) {
                this.#setBrickEmpty(brickAddress);
            } else {
                return await this.preprocess(raw.data, dtypeDescriptor);
            }
        }
        return new Uint8Array(0);
    }

    async getMaxValue(brickAddress) {
        if (!this.#maxValues[brickAddress.channel]) {
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
