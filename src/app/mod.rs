pub mod scene;
pub mod renderer;

use crate::event::handler::register_default_js_event_handlers;
use crate::event::{ChannelSettingsChange, Event, SettingsChange};
use crate::renderer::geometry::Bounds3D;
use crate::renderer::pass::present_to_screen::PresentToScreen;
use crate::renderer::pass::ray_guided_dvr::{ChannelSettings, RayGuidedDVR, Resources};
use crate::renderer::pass::{present_to_screen, ray_guided_dvr, GPUPass};
use crate::resource::sparse_residency::texture3d::SparseResidencyTexture3DOptions;
use crate::resource::VolumeManager;
use crate::util::vec::vec_equals;
use crate::volume::HtmlEventTargetVolumeDataSource;
use crate::wgsl::create_wgsl_preprocessor;
use crate::{resource, BrickedMultiResolutionMultiVolumeMeta, MultiChannelVolumeRendererSettings};
use glam::{Vec2, Vec3};
use std::sync::Arc;
use wasm_bindgen::JsCast;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, Buffer, Extent3d, SamplerDescriptor, SubmissionIndex, SurfaceConfiguration,
    TextureView,
};
use wgpu_framework::app::{GpuApp, MapToWindowEvent};
use wgpu_framework::context::{ContextDescriptor, Gpu, SurfaceContext};
use wgpu_framework::event::lifecycle::{OnCommandsSubmitted, PrepareRender, Update};
use wgpu_framework::event::window::{OnResize, OnUserEvent, OnWindowEvent};
use wgpu_framework::input::Input;
use winit::event::{
    ElementState, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::EventLoop;
use winit::platform::web::WindowExtWebSys;
use winit::window::Window;
use wgpu_framework::scene::camera::CameraView;
use crate::app::renderer::uniforms::CameraUniform;
use crate::app::scene::camera::OrbitCamera;

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
    volume_transform: glam::Mat4,
    volume_texture: VolumeManager,

    volume_render_pass: RayGuidedDVR,
    volume_render_bind_group: BindGroup,
    volume_render_global_settings_buffer: Buffer,
    volume_render_channel_settings_buffer: Buffer,
    volume_render_result_extent: Extent3d,

    present_to_screen_pass: PresentToScreen,
    present_to_screen_bind_group: BindGroup,

    channel_configuration: ChannelConfiguration,

    camera: OrbitCamera,
    resolution: Vec2,
    last_mouse_position: Vec2,
    left_mouse_pressed: bool,
    right_mouse_pressed: bool,
    settings: MultiChannelVolumeRendererSettings,
    last_channel_selection: Vec<u32>,
    channel_selection_changed: bool,
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

        let window_size = window.inner_size();
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
        let volume_texture = VolumeManager::new(
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

        let channel_mapping = volume_texture
            .get_channel_configuration(0)
            .map_channel_indices(visible_channel_indices.as_slice())
            .iter()
            .map(|c| c.unwrap() as u32)
            .collect();
        let channel_configuration = ChannelConfiguration {
            visible_channel_indices,
            channel_mapping,
        };

        let volume_render_result_extent = Extent3d {
            width: window_size.width,
            height: window_size.height,
            depth_or_array_layers: 1,
        };

        // todo: make size configurable
        let dvr_result = resource::Texture::create_storage_texture(
            gpu.device(),
            volume_render_result_extent.width,
            volume_render_result_extent.height,
        );
        // todo: the actual render pass should provide this sampler
        let volume_sampler = gpu.device().create_sampler(&SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let screen_space_sampler = gpu.device().create_sampler(&SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // todo: refactor multi-volume into scene object or whatever
        // the volume is a unit cube ([0,1]^3)
        // we translate it s.t. its center is the origin and scale it to its original dimensions
        let volume_transform = glam::Mat4::from_scale(volume_texture.normalized_volume_size())
            .mul_mat4(&glam::Mat4::from_translation(glam::Vec3::new(
                -0.5, -0.5, -0.5,
            )));
        let uniforms = ray_guided_dvr::Uniforms::default();
        let volume_render_global_settings_buffer =
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // channel settings are created for all channels s.t. the initial buffer size is large enough
        // (could also be achieved by just allocating for max visible channels -> maybe later during cleanup)
        // filtered channel settings are uploaded to gpu during update
        let channel_settings: Vec<ChannelSettings> = render_settings
            .channel_settings
            .iter()
            .map(ChannelSettings::from)
            .collect();
        let volume_render_channel_settings_buffer =
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(channel_settings.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let volume_render_pass = RayGuidedDVR::new(&volume_texture, &wgsl_preprocessor, gpu);
        let volume_render_bind_group = volume_render_pass.create_bind_group(Resources {
            volume_sampler: &volume_sampler,
            output: &dvr_result.view,
            uniforms: &volume_render_global_settings_buffer,
            channel_settings: &volume_render_channel_settings_buffer,
        });

        let present_to_screen_pass = PresentToScreen::new(gpu, surface_configuration);
        let present_to_screen_bind_group =
            present_to_screen_pass.create_bind_group(present_to_screen::Resources {
                sampler: &screen_space_sampler,
                source_texture: &dvr_result.view,
            });

        // TODO: use framework::camera instead
        // TODO: refactor these params
        let distance_from_center = 500.;
        let resolution = Vec2::new(window_size.width as f32, window_size.height as f32);
        let last_mouse_position = Vec2::new(0., 0.);
        let left_mouse_pressed = false;
        let right_mouse_pressed = false;
        let last_channel_selection = render_settings.get_sorted_visible_channel_indices();

        let camera = OrbitCamera::new(
            CameraView::new(
                Vec3::new(1., 1., 1.) * distance_from_center,

                Vec3::new(0., 0., 0.),
                Vec3::new(0., 1., 0.)
            ),
            resolution,
            0.0001,
            1000.0,
            5.0,
            0.1
        );

        Self {
            ctx: gpu.clone(),
            volume_transform,
            volume_texture,
            volume_render_pass,
            volume_render_bind_group,
            volume_render_global_settings_buffer,
            volume_render_channel_settings_buffer,
            volume_render_result_extent,
            present_to_screen_pass,
            present_to_screen_bind_group,
            channel_configuration,
            camera,
            resolution,
            last_mouse_position,
            left_mouse_pressed,
            right_mouse_pressed,
            settings: render_settings,
            last_channel_selection,
            channel_selection_changed: false,
        }
    }

    fn map_channel_settings(
        &self,
        settings: &MultiChannelVolumeRendererSettings,
    ) -> Vec<ChannelSettings> {
        let mut channel_settings = Vec::new();
        for (i, &channel) in self
            .channel_configuration
            .visible_channel_indices
            .iter()
            .enumerate()
        {
            let mut cs = ChannelSettings::from(&settings.channel_settings[channel as usize]);
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
        self.volume_render_pass.encode(
            &mut encoder,
            &self.volume_render_bind_group,
            &self.volume_render_result_extent,
        );
        self.present_to_screen_pass
            .encode(&mut encoder, &self.present_to_screen_bind_group, view);

        // todo: process request & usage buffers
        self.volume_texture
            .encode_cache_management(&mut encoder, input.frame().number());

        self.ctx.queue().submit(Some(encoder.finish()))
    }

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
        self.camera.update(input);

        let channel_settings = self.map_channel_settings(&self.settings);

        // todo: do this properly
        // a new channel selection might not have been propagated at this point -> remove some channel settings
        let mut settings = self.settings.clone();
        let mut c_settings = Vec::new();
        for &channel in self.channel_configuration.visible_channel_indices.iter() {
            c_settings.push(settings.channel_settings[channel as usize].clone());
        }
        settings.channel_settings = c_settings;

        let uniforms = ray_guided_dvr::Uniforms::new(
            CameraUniform::from(&self.camera),
            self.volume_transform,
            input.frame().number(),
            &settings,
        );

        self.ctx.queue().write_buffer(
            &self.volume_render_global_settings_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        self.ctx.queue().write_buffer(
            &self.volume_render_channel_settings_buffer,
            0,
            bytemuck::cast_slice(channel_settings.as_slice()),
        );
    }
}

impl OnCommandsSubmitted for App {
    fn on_commands_submitted(&mut self, input: &Input, _submission_index: &SubmissionIndex) {
        // todo: both of these should go into volume_texture's post_render & add_channel_configuration should not be exposed
        if self.channel_selection_changed {
            self.channel_selection_changed = false;
            let channel_mapping = self
                .volume_texture
                .add_channel_configuration(&self.last_channel_selection, input.frame().number())
                .iter()
                .map(|c| c.unwrap() as u32)
                .collect();
            self.channel_configuration = ChannelConfiguration {
                visible_channel_indices: self.last_channel_selection.clone(),
                channel_mapping,
            };
            // todo: update visible channels of octree
        }
        // todo: pass result to an octree
        let _ = self.volume_texture.update_cache(input);
    }
}

impl OnResize for App {
    fn on_resize(&mut self, _width: u32, _height: u32) {
        todo!()
    }
}
