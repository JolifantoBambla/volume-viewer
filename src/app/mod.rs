pub mod renderer;
pub mod scene;

use std::collections::HashMap;
use crate::app::renderer::dvr::common::GpuChannelSettings;
use crate::app::renderer::MultiChannelVolumeRenderer;
use crate::app::scene::volume::VolumeSceneObject;
use crate::app::scene::MultiChannelVolumeScene;
use crate::event::handler::register_default_js_event_handlers;
use crate::event::{ChannelSettingsChange, Event, SettingsChange};
use crate::resource::sparse_residency::texture3d::SparseResidencyTexture3DOptions;
use crate::resource::VolumeManager;
#[cfg(feature = "timestamp-query")]
use crate::timing::timestamp_query_helper::TimestampQueryHelper;
use crate::util::vec::vec_equals;
use crate::volume::octree::OctreeDescriptor;
use crate::volume::HtmlEventTargetVolumeDataSource;
use crate::wgsl::create_wgsl_preprocessor;
use crate::{BrickedMultiResolutionMultiVolumeMeta, MultiChannelVolumeRendererSettings};
use glam::UVec2;
use gloo_timers::callback::Interval;
use std::sync::Arc;
use wasm_bindgen::JsCast;
use web_sys::EventTarget;
use wgpu::{SubmissionIndex, SurfaceConfiguration, TextureView};
use wgpu_framework::app::{GpuApp, MapToWindowEvent};
use wgpu_framework::context::{ContextDescriptor, Gpu, SurfaceContext};
use wgpu_framework::event::lifecycle::{
    OnCommandsSubmitted, OnFrameBegin, OnFrameEnd, PrepareRender, Update,
};
use wgpu_framework::event::window::{OnResize, OnUserEvent, OnWindowEvent};
use wgpu_framework::input::Input;
use winit::event::WindowEvent;
use winit::event_loop::{EventLoop, EventLoopProxy};
use winit::platform::web::WindowExtWebSys;
use winit::window::Window;
use crate::timing::monitoring::MonitoringDataFrame;
use crate::util::extent::uvec_to_extent;

/// The `GLOBAL_EVENT_LOOP_PROXY` is a means to send data to the running application.
/// It is initialized by `start_event_loop`.
pub static mut GLOBAL_EVENT_LOOP_PROXY: Option<winit::event_loop::EventLoopProxy<Event<()>>> = None;

struct ChannelConfiguration {
    visible_channel_indices: Vec<u32>,
    channel_mapping: Vec<u32>,
}

impl ChannelConfiguration {
    #[allow(unused)]
    pub fn num_visible_channels(&self) -> usize {
        self.visible_channel_indices.len()
    }
}

pub struct App {
    ctx: Arc<Gpu>,

    scene: MultiChannelVolumeScene,
    renderer: MultiChannelVolumeRenderer,

    channel_configuration: ChannelConfiguration,

    settings: MultiChannelVolumeRendererSettings,
    last_channel_selection: Vec<u32>,
    channel_selection_changed: bool,

    #[cfg(feature = "timestamp-query")]
    render_timestamp_query_set: TimestampQueryHelper,
    #[cfg(feature = "timestamp-query")]
    octree_update_timestamp_query_set: TimestampQueryHelper,

    #[cfg(target_arch = "wasm32")]
    event_target: EventTarget,

    #[cfg(target_arch = "wasm32")]
    brick_poll_interval: Option<Interval>,

    event_loop_proxy: Option<EventLoopProxy<<Self as OnUserEvent>::UserEvent>>,
    last_frame_number: u32,
}

impl App {
    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn new(
        gpu: &Arc<Gpu>,
        window: &Window,
        surface_configuration: &SurfaceConfiguration,
        volume_meta: BrickedMultiResolutionMultiVolumeMeta,
        render_settings: MultiChannelVolumeRendererSettings,
    ) -> Self {
        let canvas = window.canvas();
        let event_target = canvas.clone().unchecked_into::<EventTarget>();

        let volume_source = Box::new(HtmlEventTargetVolumeDataSource::new(
            volume_meta,
            event_target.clone(),
        ));

        let window_size = {
            let window_size = window.inner_size();
            UVec2::new(window_size.width, window_size.height)
        };

        // todo: sort by channel importance
        // channel settings are created for all channels s.t. the initial buffer size is large enough
        // (could also be achieved by just allocating for max visible channels -> maybe later during cleanup)
        // filtered channel settings are uploaded to gpu during update
        let visible_channel_indices: Vec<u32> = render_settings
            .channel_settings
            .iter()
            .filter(|c| c.visible)
            .map(|cs| cs.channel_index)
            .collect();

        let wgsl_preprocessor = create_wgsl_preprocessor();
        let volume_manager = VolumeManager::new(
            volume_source,
            SparseResidencyTexture3DOptions {
                max_visible_channels: render_settings.create_options.max_visible_channels,
                max_resolutions: render_settings.create_options.max_resolutions,
                visible_channel_indices: visible_channel_indices.clone(),
                cache_size: uvec_to_extent(&render_settings.create_options.cache_size),
                brick_request_limit: render_settings.create_options.brick_request_limit,
                brick_transfer_limit: render_settings.create_options.brick_transfer_limit,
                ..Default::default()
            },
            &wgsl_preprocessor,
            gpu,
        );

        let channel_mapping = volume_manager
            .get_channel_configuration(0)
            .map_channel_indices(visible_channel_indices.as_slice())
            .iter()
            .map(|c| c.unwrap() as u32)
            .collect();
        let channel_configuration = ChannelConfiguration {
            visible_channel_indices,
            channel_mapping,
        };

        let volume_meta2 = volume_manager.meta().clone();
        let volume = VolumeSceneObject::new_octree_volume(
            OctreeDescriptor {
                volume: &volume_meta2,
                leaf_node_size: render_settings.create_options.leaf_node_size,
                max_num_channels: render_settings.create_options.max_visible_channels as usize,
            },
            volume_manager,
            &wgsl_preprocessor,
            gpu,
        );
        // todo: make this configurable
        //let volume = VolumeSceneObject::new_page_table_volume(volume_manager);

        let renderer = MultiChannelVolumeRenderer::new(
            window_size,
            &volume,
            &render_settings,
            &wgsl_preprocessor,
            surface_configuration,
            gpu,
        );
        let scene = MultiChannelVolumeScene::new(window_size, volume);

        let last_channel_selection = render_settings.get_sorted_visible_channel_indices();

        #[cfg(feature = "timestamp-query")]
        let render_timestamp_query_set = {
            let labels = vec![
                "DVR [begin]",
                "DVR [end]",
                "present [begin]",
                "present [end]",
                "LRU update [begin]",
                "LRU update [end]",
                "process requests [begin]",
                "process requests [end]",
            ];
            TimestampQueryHelper::new("render", labels.as_slice(), gpu)
        };

        #[cfg(feature = "timestamp-query")]
        let octree_update_timestamp_query_set = {
            let labels = vec!["octree update [begin]", "octree update [end]"];
            TimestampQueryHelper::new("render", labels.as_slice(), gpu)
        };

        Self {
            ctx: gpu.clone(),
            renderer,
            scene,
            channel_configuration,
            settings: render_settings,
            last_channel_selection,
            channel_selection_changed: false,
            #[cfg(feature = "timestamp-query")]
            render_timestamp_query_set,
            #[cfg(feature = "timestamp-query")]
            octree_update_timestamp_query_set,
            event_target,
            brick_poll_interval: None,
            event_loop_proxy: None,
            last_frame_number: 0,
        }
    }

    fn map_channel_settings(
        &self,
        settings: &MultiChannelVolumeRendererSettings,
    ) -> Vec<GpuChannelSettings> {
        let mut channel_settings = Vec::new();
        for (i, &channel) in self
            .channel_configuration
            .visible_channel_indices
            .iter()
            .enumerate()
        {
            let mut cs = GpuChannelSettings::from(&settings.channel_settings[channel as usize]);
            cs.page_table_index = self.channel_configuration.channel_mapping[i];
            channel_settings.push(cs);
        }
        channel_settings
    }
}

impl GpuApp for App {
    fn init(
        &mut self,
        window: &Window,
        event_loop: &EventLoop<Self::UserEvent>,
        _context: &SurfaceContext,
    ) {
        let canvas = window.canvas();
        register_default_js_event_handlers(&canvas, event_loop);

        // instantiate global event proxy
        unsafe {
            GLOBAL_EVENT_LOOP_PROXY = Some(event_loop.create_proxy());
        }

        self.event_loop_proxy = Some(event_loop.create_proxy());
        if let Some(event_loop_proxy) = &self.event_loop_proxy {
            event_loop_proxy.send_event(Event::PollBricks).ok();
        }
        /*
        let event_proxy = event_loop.create_proxy();
        // send a poll event every 16 ms
        self.brick_poll_interval = Some(Interval::new(16, move || {
            event_proxy.send_event(Event::PollBricks).ok();
        }));
         */
    }

    fn render(&mut self, view: &TextureView, input: &Input) -> SubmissionIndex {
        // try reading last frame's buffer
        #[cfg(feature = "timestamp-query")]
        self.render_timestamp_query_set.read_buffer();

        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut settings = self.settings.clone();

        let channel_settings = self.map_channel_settings(&settings);

        // todo: do this properly
        // a new channel selection might not have been propagated at this point -> remove some channel settings
        let mut c_settings = Vec::new();
        for &channel in self.channel_configuration.visible_channel_indices.iter() {
            c_settings.push(settings.channel_settings[channel as usize].clone());
        }
        settings.channel_settings = c_settings;

        self.renderer.render(
            view,
            &self.scene,
            &settings,
            &channel_settings,
            input,
            &mut encoder,
            #[cfg(feature = "timestamp-query")]
            &mut self.render_timestamp_query_set,
        );

        self.scene.volume_manager_mut().encode_cache_management(
            &mut encoder,
            input.frame().number(),
            #[cfg(feature = "timestamp-query")]
            &mut self.render_timestamp_query_set,
        );

        #[cfg(feature = "timestamp-query")]
        self.render_timestamp_query_set.resolve(&mut encoder);

        let submission_index = self.ctx.queue().submit(Some(encoder.finish()));

        // map this frame's buffer and prepare next frame
        #[cfg(feature = "timestamp-query")]
        self.render_timestamp_query_set.map_buffer();

        submission_index
    }

    fn get_context_descriptor() -> ContextDescriptor<'static> {
        ContextDescriptor {
            #[cfg(feature = "timestamp-query")]
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }
    }
}

impl OnWindowEvent for App {
    fn on_window_event(&mut self, _event: &WindowEvent) {}
}

impl OnUserEvent for App {
    type UserEvent = Event<()>;

    fn on_user_event(&mut self, event: &Self::UserEvent) {
        match event {
            // todo: use input & update instead
            Self::UserEvent::Window(_) => self.scene.on_user_event(event),
            Self::UserEvent::Settings(settings_change) => match settings_change {
                SettingsChange::RenderMode(mode) => {
                    self.settings.render_mode = *mode;
                }
                SettingsChange::StepScale(step_scale) => {
                    if *step_scale > 0. {
                        self.settings.step_scale = *step_scale;
                    } else {
                        log::error!("Illegal step size: {}", step_scale);
                    }
                }
                SettingsChange::MaxSteps(max_steps) => {
                    self.settings.max_steps = *max_steps;
                }
                SettingsChange::BackgroundColor(color) => {
                    self.settings.background_color = *color;
                }
                SettingsChange::ChannelSetting(channel_setting) => {
                    let i = channel_setting.channel_index as usize;
                    match channel_setting.channel_setting {
                        ChannelSettingsChange::Color(color) => {
                            self.settings.channel_settings[i].color = color;
                        }
                        ChannelSettingsChange::Visible(visible) => {
                            self.settings.channel_settings[i].visible = visible;
                        }
                        ChannelSettingsChange::Threshold(range) => {
                            self.settings.channel_settings[i].threshold_lower = range.min;
                            self.settings.channel_settings[i].threshold_upper = range.max;
                        }
                        ChannelSettingsChange::LoD(range) => {
                            self.settings.channel_settings[i].max_lod = range.min;
                            self.settings.channel_settings[i].min_lod = range.max;
                        }
                        ChannelSettingsChange::LoDFactor(lod_factor) => {
                            self.settings.channel_settings[i].lod_factor = lod_factor;
                        }
                    }
                }
            },
            Event::Brick(brick) => {
                let brick = brick.clone();
                self.scene.volume_mut().volume_manager_mut().source_mut().enqueue_brick(brick);
            }
            Event::PollBricks => {
                log::warn!("poll bricks");
                self.scene.volume_mut().update_cache(
                    self.last_frame_number,
                    #[cfg(feature = "timestamp-query")]
                    &mut self.octree_update_timestamp_query_set,
                );

                if let Some(event_loop_proxy) = &self.event_loop_proxy {
                    let proxy = event_loop_proxy.clone();
                    gloo_timers::callback::Timeout::new(16, move || {
                        proxy.send_event(Event::PollBricks).ok();
                    }).forget();
                } else {
                    log::warn!("no event loop proxy");
                }
            }
            _ => {}
        }
    }
}

impl MapToWindowEvent for App {
    fn map_to_window_event(&self, user_event: &Self::UserEvent) -> Option<WindowEvent> {
        match user_event {
            Self::UserEvent::Window(e) => Some(e.clone()),
            _ => None,
        }
    }
}

impl OnFrameBegin for App {
    fn on_frame_begin(&mut self) {
        // todo: dispatch canvas event
    }
}

impl PrepareRender for App {
    fn prepare_render(&mut self, _input: &Input) {
        let channel_selection = self.settings.get_sorted_visible_channel_indices();

        if !vec_equals(&channel_selection, &self.last_channel_selection) {
            self.last_channel_selection = channel_selection;
            self.channel_selection_changed = true;
        };
    }
}

impl Update for App {
    fn update(&mut self, input: &Input) {
        self.scene.update(input);
    }
}

impl OnCommandsSubmitted for App {
    fn on_commands_submitted(&mut self, input: &Input, _submission_index: &SubmissionIndex) {
        // todo: both of these should go into volume_texture's post_render & add_channel_configuration should not be exposed
        if self.channel_selection_changed {
            self.channel_selection_changed = false;
            let channel_mapping = self
                .scene
                .volume_mut()
                .update_channel_selection(&self.last_channel_selection, input.frame().number());
            self.channel_configuration = ChannelConfiguration {
                visible_channel_indices: self.last_channel_selection.clone(),
                channel_mapping,
            };
        }
        self.scene.volume_mut().update_cache_meta(input);
        self.last_frame_number = input.frame().number();
    }
}

impl OnFrameEnd for App {
    fn on_frame_end(&mut self, _input: &Input) {
        // todo: dispatch frame end canvas event

        #[cfg(feature = "timestamp-query")]
        {
            let mut monitoring = MonitoringDataFrame::default();
            if let Some(diff) = self.render_timestamp_query_set.get_last_diff("DVR [begin]", "DVR [end]") {
                let duration = diff as f64 / 1_000_000.0;
                monitoring.dvr = duration;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("DVR [begin]") {
                monitoring.dvr_begin = timestamp as f64 / 1_000_000.0;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("DVR [end]") {
                monitoring.dvr_end = timestamp as f64 / 1_000_000.0;
            }
            if let Some(diff) = self.render_timestamp_query_set.get_last_diff("present [begin]", "present [end]") {
                let duration = diff as f64 / 1_000_000.0;
                monitoring.present = duration;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("present [begin]") {
                monitoring.present_begin = timestamp as f64 / 1_000_000.0;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("present [end]") {
                monitoring.present_end = timestamp as f64 / 1_000_000.0;
            }
            if let Some(diff) = self.render_timestamp_query_set.get_last_diff("LRU update [begin]", "LRU update [end]") {
                let duration = diff as f64 / 1_000_000.0;
                monitoring.lru_update = duration;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("LRU update [begin]") {
                monitoring.lru_update_begin = timestamp as f64 / 1_000_000.0;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("LRU update [end]") {
                monitoring.lru_update_end = timestamp as f64 / 1_000_000.0;
            }
            if let Some(diff) = self.render_timestamp_query_set.get_last_diff("process requests [begin]", "process requests [end]") {
                let duration = diff as f64 / 1_000_000.0;
                monitoring.process_requests = duration;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("process requests [begin]") {
                monitoring.process_requests_begin = timestamp as f64 / 1_000_000.0;
            }
            if let Some(&timestamp) = self.render_timestamp_query_set.get_last("process requests [end]") {
                monitoring.process_requests_end = timestamp as f64 / 1_000_000.0;
            }
            if let Some(diff) = self.octree_update_timestamp_query_set.get_last_diff("octree update [begin]", "octree update [end]") {
                let duration = diff as f64 / 1_000_000.0;
                monitoring.octree_update = duration;

                if let Some(&timestamp) = self.octree_update_timestamp_query_set.get_last("octree update [begin]") {
                    monitoring.octree_update_begin = timestamp as f64 / 1_000_000.0;
                }
                if let Some(&timestamp) = self.octree_update_timestamp_query_set.get_last("octree update [end]") {
                    monitoring.octree_update_end = timestamp as f64 / 1_000_000.0;
                }
                log::info!("octree update took {:?} ms", monitoring.octree_update);
            }
            #[cfg(target_arch = "wasm32")]
            {
                let mut event_data: HashMap<&str, MonitoringDataFrame> = HashMap::new();
                event_data.insert("monitoring", monitoring);
                let monitoring_event = web_sys::CustomEvent::new("monitoring").ok().unwrap();
                monitoring_event.init_custom_event_with_can_bubble_and_cancelable_and_detail(
                    "monitoring",
                    false,
                    false,
                    &serde_wasm_bindgen::to_value(&event_data).expect("Could not serialize monitoring event data"),
                );
                self.event_target.dispatch_event(&monitoring_event).ok();
            }
            #[cfg(not(target_arch = "wasm32"))]
            log::info!("Monitoring: {:?}", monitoring);
        }
    }
}

impl OnResize for App {
    fn on_resize(&mut self, _width: u32, _height: u32) {
        todo!()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(interval) = self.brick_poll_interval.take() {
            interval.cancel();
        }
    }
}
