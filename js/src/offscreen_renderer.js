import * as Comlink from '../../node_modules/comlink/dist/esm/comlink.mjs';

import { toWrappedEvent } from './event.js';
import {
    ChannelSettings,
    VolumeRendererSettings,
    RENDER_MODE_OCTREE,
    RENDER_MODE_PAGE_TABLE
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
                Octree: RENDER_MODE_OCTREE,
                "Page Table": RENDER_MODE_PAGE_TABLE
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

    const monitoringParams = {
        dvr: 0.0,
        present: 0.0,
        lruUpdate: 0.0,
        processRequests: 0.0,
        octree_update: 0.0,
    };
    const monitoringFolder = pane.addFolder({
        title: 'Monitoring'
    });
    const dvrFpsGraph = monitoringFolder.addBlade({
        view: 'fpsgraph',
        label: 'DVR (FPS)',
    });
    monitoringFolder.addMonitor(monitoringParams, 'dvr', {
        label: 'DVR (ms [0,32])',
        view: 'graph',
        max: 32,
    });
    const presentFpsGraph = monitoringFolder.addBlade({
        view: 'fpsgraph',
        label: 'Present (FPS)',
    });
    monitoringFolder.addMonitor(monitoringParams, 'present', {
        label: 'Present (ms [0,5])',
        view: 'graph',
        max: 5,
    });
    const lruUpdateFpsGraph = monitoringFolder.addBlade({
        view: 'fpsgraph',
        label: 'LRU update (FPS)',
    });
    monitoringFolder.addMonitor(monitoringParams, 'lruUpdate', {
        label: 'LRU update (ms [0, 0.2])',
        view: 'graph',
        max: 0.2,
    });
    const processRequestsFpsGraph = monitoringFolder.addBlade({
        view: 'fpsgraph',
        label: 'Process requests (FPS)',
    });
    monitoringFolder.addMonitor(monitoringParams, 'processRequests', {
        label: 'Process requests (ms [0, 0.2])',
        view: 'graph',
        max: 0.2,
    });
    /*
    const octreeUpdateFpsGraph = monitoringFolder.addBlade({
        view: 'fpsgraph',
        label: 'Octree update (FPS)',
    });
    */
    monitoringFolder.addMonitor(monitoringParams, 'octree_update', {
        label: 'Octree update (ms [0, 0.1])',
        view: 'graph',
        max: 0.1,
    });

    const onMonitoringDataFrame = (monitoring) => {
        monitoringParams.dvr = monitoring.dvr;
        monitoringParams.present = monitoring.present;
        monitoringParams.lruUpdate = monitoring.lruUpdate;
        monitoringParams.processRequests = monitoring.processRequests;
        monitoringParams.octree_update = monitoring.octree_update;

        dvrFpsGraph.controller_.valueController.stopwatch_.begin({getTime: () => monitoring.dvrBegin});
        dvrFpsGraph.controller_.valueController.stopwatch_.end({getTime: () => monitoring.dvrEnd});
        presentFpsGraph.controller_.valueController.stopwatch_.begin({getTime: () => monitoring.presentBegin});
        presentFpsGraph.controller_.valueController.stopwatch_.end({getTime: () => monitoring.presentEnd});
        lruUpdateFpsGraph.controller_.valueController.stopwatch_.begin({getTime: () => monitoring.lruUpdateBegin});
        lruUpdateFpsGraph.controller_.valueController.stopwatch_.end({getTime: () => monitoring.lruUpdateEnd});
        processRequestsFpsGraph.controller_.valueController.stopwatch_.begin({getTime: () => monitoring.processRequestsBegin});
        processRequestsFpsGraph.controller_.valueController.stopwatch_.end({getTime: () => monitoring.processRequestsEnd});
        /*
        if (monitoring.octree_update > 0.0) {
            octreeUpdateFpsGraph.controller_.valueController.stopwatch_.begin({getTime: () => monitoring.octreeUpdateBegin});
            octreeUpdateFpsGraph.controller_.valueController.stopwatch_.end({getTime: () => monitoring.octreeUpdateEnd});
        }
         */
    };

    /*
    const exportSettingsButton = pane.addButton({
        title: 'Export Settings',
    });
    exportSettingsButton.on('click', function() {
        const preset = pane.exportPreset();
        console.log(preset);
    });

     */

    return [volumeRendererSettings, onMonitoringDataFrame];
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

        const [volumeRendererSettings, onMonitoringDataFrame] = await createUI(offscreenRenderer, config);

        worker.addEventListener('message', e => {
            if (e.data.type === 'monitoring') {
                onMonitoringDataFrame(e.data.data);
            }
        })

        return {
            offscreenRenderer,
            volumeRendererSettings,
        };
    } else {
        throw Error(`Canvas with id "${config.canvasId}" does not support offscreen rendering.`);
    }
}
