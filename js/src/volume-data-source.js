import init, {
    initThreadPool,
    initPreprocessing,
    clampU8, clampU16, clampF32,
    isZeroU8, isZeroU16, isZeroF32,
    minU8, minU16, minF32,
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

const minValue = (data, {type = 'u1'}) => {
    switch (type) {
        case 'u1':
            return minU8(data);
        case 'u2':
            return minU16(data);
        case 'f4':
            return minF32(data);
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
    }
}

const maxValue = (data, {type = 'u1'}) => {
    switch (type) {
        case 'u1':
            return maxU8(data);
        case 'u2':
            return maxU16(data);
        case 'f4':
            return maxF32(data);
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
    }
}

const getMaxValueForDataTypeDescriptor = ({type = 'u1'}) => {
    switch (type) {
        case 'u1':
            return 255.;
        case 'u2':
            return 65535.;
        case 'f4':
            return 65535.;//Number.MAX_VALUE;
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
    }
};

const isZero = (data, {type = 'u1'}, threshold) => {
    switch (type) {
        case 'u1':
            return isZeroU8(data, threshold);
        case 'u2':
            return isZeroU16(data, threshold);
        case 'f4':
            return isZeroF32(data, threshold);
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
    }
};

const clamp = (data, thresholdLower, thresholdUpper, {type = 'u1'}) => {
    switch (type) {
        case 'u1':
            return clampU8(data, thresholdLower, thresholdUpper);
        case 'u2':
            return clampU16(data, thresholdLower, thresholdUpper);
        case 'f4':
            return clampU8(data, thresholdLower, thresholdUpper);//clampF32(data, thresholdLower, thresholdUpper);
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
    }
};

// todo: this actually scales - should add a simple cast as well
const castToU8 = (data, {type = 'u1'}) => {
    switch (type) {
        case 'u1':
            return data;
        case 'u2':
            return scaleU16ToU8(data, getMaxValueForDataTypeDescriptor({type}));
        case 'f4':
            return new Uint8Array(scaleF32ToU8(data, getMaxValueForDataTypeDescriptor({type})));
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
    }
};

const logTransformToU8 = (data, {type = 'u1'}) => {
    const maxValue = getMaxValueForDataTypeDescriptor({type});
    switch (type) {
        case 'u1':
            return logScaleU8(data, maxValue);
        case 'u2':
            return logScaleU16(data, maxValue);
        case 'f4':
            return logScaleF32(data, maxValue);
        default:
            throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
    }
};

export const BRICK_REQUEST_EVENT = 'data-loader:brick-request';
export const BRICK_RESPONSE_EVENT = 'data-loader:brick-response';

export const PREPROCESS_METHOD_CAST = 'cast';
export const PREPROCESS_METHOD_SCALE_TO_MAX = 'scaleToMax';
export const PREPROCESS_METHOD_LOG = 'log';
export const PREPROCESS_METHOD_LOG_SCALE = 'logScale';
export const PREPROCESS_METHODS = [
    PREPROCESS_METHOD_CAST,
    PREPROCESS_METHOD_SCALE_TO_MAX,
    PREPROCESS_METHOD_LOG,
    PREPROCESS_METHOD_LOG_SCALE
];

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
    }
}

export class ChannelInfo {
    name;
    constructor(name) {
        this.name = name;
    }

}

export class MultiResolutionVolumeMeta {
    constructor(brickSize, scale, resolutions, channels) {
        this.brickSize = brickSize;
        this.scale = scale;
        this.resolutions = resolutions;
        this.channels = channels;
    }
}

export class RawVolumeChunk {
    data;
    shape;
    chunk;
}

export class Brick {
    data;
    min;
    max;
}

/**
 * An interface for brick loading.
 */
export class BrickLoader {
    /**
     * Loads a chunk of the volume at a given resolution level
     * @returns {Promise<Brick>}
     * @param brickAddress
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

    async getMaxValue(brickAddress, {type = 'u1'}) {
        throw new Error("not implemented");
    }

    get isZeroThreshold() {
        return this.#config.preprocessing.isZeroThreshold || 0.0;
    }

    get preprocessingMethod() {
        return this.#config.preprocessing.preprocessingMethod || PREPROCESS_METHOD_CAST;
    }

    async #scaleToMax(data, {type = 'u1'}, brickAddress) {
        const maxValue = await this.getMaxValue(brickAddress, {type});
        switch (type) {
            case 'u1':
                return scaleU8ToU8(data, maxValue);
            case 'u2':
                return scaleU16ToU8(data, maxValue);
            case 'f4':
                return new Uint8Array(scaleF32ToU8(data, maxValue));
            default:
                throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
        }
    }

    async #logScale(data, {type = 'u1'}, brickAddress) {
        const maxValue = await this.getMaxValue(brickAddress, {type});
        switch (type) {
            case 'u1':
                return logScaleU8(data, maxValue);
            case 'u2':
                return logScaleU16(data, maxValue);
            case 'f4':
                return logScaleF32(data, maxValue);
            default:
                throw Error(`Expected one of ['u1', 'u2', 'f4'], got ${type}`);
        }
    }

    async preprocess(data, {type = 'u1'}, brickAddress) {
        switch (this.preprocessingMethod) {
            case PREPROCESS_METHOD_LOG_SCALE:
                return this.#logScale(data, {type}, brickAddress);
            case PREPROCESS_METHOD_SCALE_TO_MAX:
                return this.#scaleToMax(data, {type}, brickAddress);
            case PREPROCESS_METHOD_LOG:
                return logTransformToU8(data, {type});
            case PREPROCESS_METHOD_CAST:
            default:
                return castToU8(data, {type});
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

        this.#zarrGroup = zarrGroup;
        this.#omeZarrMeta = omeZarrMeta;
        this.#zarrArrays = zarrArrays;

        const numChannels = this.#zarrArrays[0].meta.shape[this.#zarrArrays[0].meta.shape.length - 4];

        const brickSize = config.bricks.minimumSize;
        for (const arr of this.#zarrArrays) {
            for (let i = 0; i < 3; ++i) {
                brickSize[i] = Math.min(
                    config.bricks.maximumSize[i],
                    arr.shape[arr.shape.length - 1 - i],
                    Math.max(
                        config.bricks.minimumSize[i],
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

        this.#volumeMeta = new MultiResolutionVolumeMeta(brickSize, scale, resolutions, [...new Array(numChannels).keys()].map(k => new ChannelInfo(`${k}`)));
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
            const data = clamp(
                await this.preprocess(raw.data, dtypeDescriptor, brickAddress),
                this.isZeroThreshold,
                1.0,
                dtypeDescriptor
            );

            if (isZero(data, { type: 'u1' }, this.isZeroThreshold)) {
                this.#setBrickEmpty(brickAddress);
            } else {
                return {
                    data: typeof data === Uint8Array ? data : new Uint8Array(data),
                    min: 0,  //minU8(data),
                    max: 255 //maxU8(data),
                };
            }
        }
        return {
            data: new Uint8Array(0),
            min: 0,
            max: 0
        };
    }

    async getMaxValue(brickAddress, {type = 'u1'}) {
        if (!this.#maxValues[brickAddress.channel]) {
            this.#maxValues[brickAddress.channel] = maxValue(
                (await this.#zarrArrays
                        .slice()
                        .reverse()[0]
                        .getRaw([0, brickAddress.channel])
                ).data,
                {type}
            );
        }
        return this.#maxValues[brickAddress.channel];
    }
}

// -------------------- CONFIG ----------------------------
export class DataStoreConfig {
    store;
    path;
    sourceType;

    constructor({store, path, sourceType = "OME-Zarr"}) {
        this.store = store;
        this.path = path;
        this.sourceType = sourceType;
    }
}

export class PreprocessConfig {
    isZeroThreshold;
    preprocessingMethod;

    constructor({preprocessingMethod = PREPROCESS_METHOD_CAST, isZeroThreshold = 0.0}) {
        if (!PREPROCESS_METHODS.includes(preprocessingMethod)) {
            throw Error(`Expected one of ${PREPROCESS_METHODS}, got ${preprocessingMethod}`);
        }
        if (typeof isZeroThreshold !== 'number' || isZeroThreshold < 0.0) {
            throw Error(`Expected a number >= 0.0, got ${isZeroThreshold}`);
        }
        this.preprocessingMethod = preprocessingMethod;
        this.isZeroThreshold = isZeroThreshold;
    }
}

export class PageTableConfig {
    minimumSize;
    maximumSize;

    constructor({ maxSize = [256, 256, 256], minSize = [32, 32, 32] }) {
        if (maxSize.length !== 3 || minSize.length !== 3) {
            throw Error(`Expected brick sizes of exactly length 3, got ${minSize} and ${maxSize}`);
        }
        for (let i = 0; i < 3; ++i) {
            if (maxSize[i] < minSize[i]) {
                throw Error(`Expected minimum size to be component wise <= minimum size, got ${minSize} and ${maxSize}`);
            }
        }
        this.minimumSize = minSize;
        this.maximumSize = maxSize;
    }
}

export class VolumeDataSourceConfig {
    dataStore;
    preprocessing;
    bricks;

    constructor(dataStore, preprocessConfig, brickConfig) {
        this.dataStore = dataStore;
        this.preprocessing = preprocessConfig;
        this.bricks = brickConfig;
    }
}

// -------------------- CONFIG ----------------------------

async function createVolumeDataSource(config) {
    const mode = 'r';

    const {store, path} = config.dataStore;
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
        initPreprocessing();

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
                        brick: await this.#dataSource.loadBrick(brickAddress),
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
