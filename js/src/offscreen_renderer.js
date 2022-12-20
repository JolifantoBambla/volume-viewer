import * as Comlink from '../../node_modules/comlink/dist/esm/comlink.mjs';

import { toWrappedEvent } from './event.js';
import {
    ChannelSettings,
    VolumeRendererSettings,
    RENDER_MODE_DIRECT
} from './volume-renderer.js';

export class Config {
    constructor({canvas, numThreads, logging, dataSource, rendererCreateOptions}) {
        this.canvas = canvas || {
            width: 800,
            height: 600,
        };
        this.numThreads = numThreads || navigator.hardwareConcurrency;
        this.logging = logging || 'error';
        this.dataSource = dataSource;
        if (!this.dataSource) {
            throw new Error("No datasource given!");
        }
        this.rendererCreateOptions = rendererCreateOptions;
    }
}

function getOrCreateCanvas(canvasConfig) {
    if (canvasConfig.canvasId) {
        return document.querySelector(canvasConfig.canvasId);
    } else {
        const parent = canvasConfig.parentId ? document.querySelector(canvasConfig.parentId) : document.body;
        const canvas = document.createElement('canvas');
        canvas.width = canvasConfig.width;
        canvas.height = canvasConfig.height;
        parent.appendChild(canvas);
        return canvas;
    }
}

async function createUI(offscreenRenderer, config) {
    const volumeMetaData = await offscreenRenderer.volumeMeta;
    const dispatchGlobalSettingsChange = e => {
        offscreenRenderer.dispatchUIEvent(JSON.stringify({
            type: 'rendersettings',
            setting: e.presetKey,
            value: e.value,
        }));
    };
    const dispatchChannelSettingsChange = (e, channelIndex) => {
        const channelSetting = {};
        channelSetting[`${e.presetKey}`] = e.value;
        offscreenRenderer.dispatchUIEvent(JSON.stringify({
            type: 'rendersettings',
            setting: 'channelSetting',
            value: {
                channelIndex,
                channelSetting,
            },
        }));
    }

    const volumeRendererSettings = new VolumeRendererSettings({
        createOptions: config.rendererCreateOptions,
        channelSettings: volumeMetaData.channels.map((_, channelIndex) => {
            return new ChannelSettings({
                channelIndex,
                visible: channelIndex === 0,
                minLoD: volumeMetaData.resolutions.length - 1,
            });
        }),
    });

    // TODO: clean this up
    // can it do transfer functions?
    // maybe: https://github.com/tweakpane/plugin-essentials
    const pane = new Tweakpane.Pane({
        title: 'Settings',
        expanded: true,
    });
    pane.registerPlugin(TweakpaneEssentialsPlugin);

    const renderPane = pane.addFolder({
        title: 'Rendering',
        expanded: true,
    });
    const renderModeSelector = renderPane.addInput(
        volumeRendererSettings, 'renderMode',
        {
            options: {
                Direct: RENDER_MODE_DIRECT,
            }
        }
    );
    const stepsSlider = renderPane.addInput(
        volumeRendererSettings, 'stepScale',
        {min: 0.5, max: 10.0, step: 0.2, label: 'Step spacing'}
    );
    const maxStepsSlider = renderPane.addInput(
        volumeRendererSettings, 'maxSteps',
        {min: 1, max: 1000, step: 1, label: 'Max. steps'}
    );
    const backgroundColorPicker = renderPane.addInput(volumeRendererSettings, 'backgroundColor', {
        label: 'Background color',
        picker: 'inline',
        expanded: false,
        color: {type: 'float'},
    });
    backgroundColorPicker.on('change', e => {
        const cToHex = c => {
            const hex = Math.round(c * 255).toString(16);
            return hex.length === 1 ? `0${hex}` : hex;
        }
        function rgbToHex({r, g, b}) {
            return `#${cToHex(r)}${cToHex(g)}${cToHex(b)}`
        }
        dispatchGlobalSettingsChange(e);
        document.body.style.backgroundColor = rgbToHex(e.value);
    });

    [renderModeSelector, stepsSlider, maxStepsSlider, backgroundColorPicker]
        .forEach(i => i.on('change', dispatchGlobalSettingsChange));

    const channelsSettings = renderPane.addFolder({
        title: 'Channel Settings',
        expanded: true,
    });
    volumeRendererSettings.channelSettings.forEach(c => {
        const channelSettings = channelsSettings.addFolder({
            title: c.channelName,
            expanded: c.visible,
        });
        const visibleToggle = channelSettings.addInput(c, 'visible');
        const colorPicker = channelSettings.addInput(c, 'color', {
            picker: 'inline',
            expanded: c.visible,
            color: {type: 'float'},
        });
        const lodSlider = channelSettings.addInput({lod: {min: c.maxLoD, max: c.minLoD}}, 'lod', {
            label: 'Resolution Levels',
            min: 0, max: c.minLoD, step: 1,
        });
        const lodFactorSlider = channelSettings.addInput(c, 'lodFactor', {
            label: 'LoD Factor',
            min: 0.001, max: 2.0, step: 0.001
        })
        const thresholdSlider = channelSettings.addInput({threshold: {min: c.thresholdLower, max: c.thresholdUpper}}, 'threshold', {
            label: 'Threshold',
            min: 0.0, max: 1.0, step: 0.01,
        });

        [colorPicker, visibleToggle, lodSlider, lodFactorSlider, thresholdSlider]
            .forEach(i => i.on('change', e => dispatchChannelSettingsChange(e, c.channelIndex)));
    });

    const exportSettingsButton = pane.addButton({
        title: 'Export Settings',
    });
    exportSettingsButton.on('click', function() {
        const preset = pane.exportPreset();
        console.log(preset);
    });

    return volumeRendererSettings;
}

export async function createOffscreenRenderer(config) {
    const canvas = getOrCreateCanvas(config.canvas);

    const hasOffscreenSupport = !!canvas.transferControlToOffscreen;
    if (hasOffscreenSupport) {
        // instantiate worker
        const worker = new Worker('./js/src/volume-renderer-thread.js', { type: "module" });
        const offscreenRenderer = Comlink.wrap(worker);

        // transfer control over canvas to worker
        const offscreen = canvas.transferControlToOffscreen();
        Comlink.transfer(offscreen, [offscreen]);
        await offscreenRenderer.initialize(offscreen, config);

        // register UI event listeners
        const dispatchToWorker = (e) => {
            e.preventDefault();
            offscreenRenderer.dispatchCanvasEvent(JSON.stringify(toWrappedEvent(e)));
        };
        canvas.onmousedown = dispatchToWorker;
        canvas.onmouseup = dispatchToWorker;
        canvas.onmousemove = dispatchToWorker;
        canvas.onwheel = dispatchToWorker;
        window.onkeydown = dispatchToWorker;
        window.onkeyup = dispatchToWorker;
        window.onkeypress = dispatchToWorker;

        const volumeRendererSettings = await createUI(offscreenRenderer, config);

        return {
            offscreenRenderer,
            volumeRendererSettings,
        };
    } else {
        throw Error(`Canvas with id "${config.canvasId}" does not support offscreen rendering.`);
    }
}
