pub mod renderer;
pub mod scene;

use std::collections::{HashMap, VecDeque};
use crate::app::renderer::dvr::common::GpuChannelSettings;
use crate::app::renderer::MultiChannelVolumeRenderer;
use crate::app::scene::volume::VolumeSceneObject;
use crate::app::scene::MultiChannelVolumeScene;
use crate::event::handler::register_default_js_event_handlers;
use crate::event::{ChannelSettingsChange, Event, SettingsChange};
use crate::resource::sparse_residency::texture3d::SparseResidencyTexture3DOptions;
use crate::resource::VolumeManager;
use crate::util::vec::vec_equals;
use crate::volume::octree::OctreeDescriptor;
use crate::volume::HtmlEventTargetVolumeDataSource;
use crate::wgsl::create_wgsl_preprocessor;
use crate::{BrickedMultiResolutionMultiVolumeMeta, MultiChannelVolumeRendererSettings};
use glam::UVec2;
use std::sync::Arc;
use wasm_bindgen::JsCast;
#[cfg(feature = "timestamp-query")]
use wgpu::MapMode;
use wgpu::{CommandEncoder, SubmissionIndex, SurfaceConfiguration, TextureView};
use wgpu_framework::app::{GpuApp, MapToWindowEvent};
use wgpu_framework::context::{ContextDescriptor, Gpu, SurfaceContext};
use wgpu_framework::event::lifecycle::{
    OnCommandsSubmitted, OnFrameBegin, OnFrameEnd, PrepareRender, Update,
};
use wgpu_framework::event::window::{OnResize, OnUserEvent, OnWindowEvent};
#[cfg(feature = "timestamp-query")]
use wgpu_framework::gpu::query_set::TimeStampQuerySet;
use wgpu_framework::input::Input;
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
use winit::platform::web::WindowExtWebSys;
use winit::window::Window;
use wgpu_framework::gpu::buffer::MappableBuffer;

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

#[cfg(feature = "timestamp-query")]
pub struct TimestampQueryHelper {
    gpu: Arc<Gpu>,
    query_set: TimeStampQuerySet,
    labels: Vec<String>,
    timings: HashMap<String, Vec<u64>>,
    resolve_buffer_pool: Vec<MappableBuffer<u64>>,
    mapped_buffer_queue: VecDeque<MappableBuffer<u64>>,
    buffer_in_last_submit: Option<MappableBuffer<u64>>,
}

#[cfg(feature = "timestamp-query")]
impl TimestampQueryHelper {
    pub fn new(label: &str, labels: &[&str], gpu: &Arc<Gpu>) -> Self {
        let query_set = TimeStampQuerySet::from_labels(label, labels, gpu);
        let mut own_labels = Vec::new();
        let mut timings = HashMap::new();
        for l in labels {
            own_labels.push(l.to_string());
            timings.insert(l.to_string(), Vec::new());
        }

        Self {
            gpu: gpu.clone(),
            query_set,
            labels: own_labels,
            timings,
            resolve_buffer_pool: Vec::new(),
            mapped_buffer_queue: VecDeque::new(),
            buffer_in_last_submit: None,
        }
    }

    pub fn write_timestamp(&mut self, command_encoder: &mut CommandEncoder) {
        if let Err(error) = self.query_set.write_timestamp(command_encoder) {
            log::error!("could not write timestamp: {}", error);
        };
    }

    pub fn resolve(&mut self, command_encoder: &mut CommandEncoder) {
        if self.buffer_in_last_submit.is_some() {
            panic!("last submit's buffer not processed!");
        }
        let buffer = self.resolve_buffer_pool.pop().unwrap_or_else(|| self.query_set.create_resolve_buffer(&self.gpu));
        if let Err(error) = self.query_set.resolve(command_encoder, &buffer) {
            log::error!("could not resolve timestamp query set: {}", error);
        }
        self.buffer_in_last_submit = Some(buffer);
    }

    pub fn map_buffer(&mut self) {
        if let Some(buffer) = self.buffer_in_last_submit.take() {
            buffer.map_async(MapMode::Read, ..).expect("Could not map resolve buffer");
            self.mapped_buffer_queue.push_front(buffer);
        }
    }

    pub fn read_buffer(&mut self) {
        let has_mapped_buffer = if let Some(buffer) = self.mapped_buffer_queue.back() {
            buffer.is_mapped()
        } else {
            false
        };
        if has_mapped_buffer {
            let buffer = self.mapped_buffer_queue.pop_back().unwrap();
            let timestamps = buffer.read_all().expect("Could not read mapped buffer");
            for (i, label) in self.labels.iter().enumerate() {
                self.timings.get_mut(label).unwrap().push(*timestamps.get(i).unwrap());
            }
            self.resolve_buffer_pool.push(buffer);
        }
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

    //#[cfg(feature = "timestamp-query")]
    //octree_update_timestamp_query_set: TimeStampQuerySet,
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

        let volume_source = Box::new(HtmlEventTargetVolumeDataSource::new(
            volume_meta,
            canvas.unchecked_into::<web_sys::EventTarget>(),
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

        #[cfg(feature = "timestamp-query")]
        let render_timestamp_query_set = TimestampQueryHelper::new(
        "render",
        labels.as_slice(),
        gpu,
        );

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
    }

    fn render(&mut self, view: &TextureView, input: &Input) -> SubmissionIndex {
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

        // todo: read & store results
        #[cfg(feature = "timestamp-query")]
        let mut time_stamp_query_set = TimeStampQuerySet::from_labels(
            "render",
            labels.as_slice(),
            &self.ctx,
        );

        #[cfg(feature = "timestamp-query")]
        let query_read_buffer = time_stamp_query_set.create_resolve_buffer(&self.ctx);

        self.renderer.render(
            view,
            &self.scene,
            &settings,
            &channel_settings,
            input,
            &mut encoder,
            #[cfg(feature = "timestamp-query")]
            &mut time_stamp_query_set,
        );

        self.scene.volume_manager_mut().encode_cache_management(
            &mut encoder,
            input.frame().number(),
            #[cfg(feature = "timestamp-query")]
            &mut time_stamp_query_set,
        );

        #[cfg(feature = "timestamp-query")]
        time_stamp_query_set
            .resolve(&mut encoder, &query_read_buffer)
            .expect("Encountered incompatible buffer for timestamp query set");

        let submission_index = self.ctx.queue().submit(Some(encoder.finish()));

        #[cfg(feature = "timestamp-query")]
        query_read_buffer
            .map_async(MapMode::Read, ..)
            .expect("Could not map buffer");

        submission_index
    }

    #[cfg(feature = "timestamp-query")]
    fn get_context_descriptor() -> ContextDescriptor<'static> {
        log::info!("timestamp query!!");
        ContextDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }
    }

    #[cfg(not(feature = "timestamp-query"))]
    fn get_context_descriptor() -> ContextDescriptor<'static> {
        ContextDescriptor::default()
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
        self.scene.volume_mut().update_cache(input);
    }
}

impl OnFrameEnd for App {
    fn on_frame_end(&mut self, _input: &Input) {
        // todo: dispatch frame end canvas event
    }
}

impl OnResize for App {
    fn on_resize(&mut self, _width: u32, _height: u32) {
        todo!()
    }
}
