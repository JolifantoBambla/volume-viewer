pub mod common;
pub mod dvr;

use crate::app::renderer::common::CameraUniform;
use crate::app::renderer::dvr::common::{GpuChannelSettings, Uniforms};
use crate::app::renderer::dvr::{RayGuidedDVR, Resources};
use crate::app::scene::MultiChannelVolumeScene;
use crate::renderer::pass::present_to_screen::PresentToScreen;
use crate::renderer::pass::{present_to_screen, GPUPass};
use glam::UVec2;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, Buffer, CommandEncoder, Extent3d, SamplerDescriptor, SurfaceConfiguration,
    TextureView,
};
use wgpu_framework::context::Gpu;
use wgpu_framework::input::Input;
use wgsl_preprocessor::WGSLPreprocessor;
use crate::app::scene::volume::VolumeSceneObject;
use crate::{MultiChannelVolumeRendererSettings, resource};
use crate::app::renderer::dvr::page_table::PageTableDVR;

#[derive(Debug)]
pub struct MultiChannelVolumeRenderer {
    gpu: Arc<Gpu>,
    volume_render_pass: RayGuidedDVR,
    volume_render_bind_group: BindGroup,
    volume_render_global_settings_buffer: Buffer,
    volume_render_channel_settings_buffer: Buffer,
    volume_render_result_extent: Extent3d,
    present_to_screen_pass: PresentToScreen,
    present_to_screen_bind_group: BindGroup,

    // to switch between the two render modes for debugging
    page_table_render_pass: PageTableDVR,
    page_table_bind_group: BindGroup,
}

impl MultiChannelVolumeRenderer {
    pub fn new(
        window_size: UVec2,
        volume: &VolumeSceneObject,
        render_settings: &MultiChannelVolumeRendererSettings,
        wgsl_preprocessor: &WGSLPreprocessor,
        surface_configuration: &SurfaceConfiguration,
        gpu: &Arc<Gpu>,
    ) -> Self {
        let volume_render_result_extent = Extent3d {
            width: window_size.x,
            height: window_size.y,
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

        let uniforms = Uniforms::default();
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
        let channel_settings: Vec<GpuChannelSettings> = render_settings
            .channel_settings
            .iter()
            .map(GpuChannelSettings::from)
            .collect();
        let volume_render_channel_settings_buffer =
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(channel_settings.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let volume_render_pass = RayGuidedDVR::new(volume, wgsl_preprocessor, gpu);
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

        let page_table_render_pass = PageTableDVR::new(volume.volume_manager(), wgsl_preprocessor, gpu);
        let page_table_bind_group = page_table_render_pass.create_bind_group(Resources {
            volume_sampler: &volume_sampler,
            output: &dvr_result.view,
            uniforms: &volume_render_global_settings_buffer,
            channel_settings: &volume_render_channel_settings_buffer,
        });

        Self {
            gpu: gpu.clone(),
            volume_render_pass,
            volume_render_bind_group,
            volume_render_global_settings_buffer,
            volume_render_channel_settings_buffer,
            volume_render_result_extent,
            present_to_screen_pass,
            present_to_screen_bind_group,

            page_table_render_pass,
            page_table_bind_group
        }
    }

    pub fn render(
        &self,
        render_target: &TextureView,
        scene: &MultiChannelVolumeScene,
        settings: &MultiChannelVolumeRendererSettings,
        channel_settings: &Vec<GpuChannelSettings>,
        input: &Input,
        command_encoder: &mut CommandEncoder,
        force_page_table_dvr: bool,
    ) {
        let uniforms = Uniforms::new(
            CameraUniform::from(scene.camera()),
            scene.volume_transform(),
            input.frame().number(),
            settings,
        );

        self.gpu.queue().write_buffer(
            &self.volume_render_global_settings_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        self.gpu.queue().write_buffer(
            &self.volume_render_channel_settings_buffer,
            0,
            bytemuck::cast_slice(channel_settings.as_slice()),
        );

        // todo: remove (debug)
        if force_page_table_dvr {
            self.page_table_render_pass.encode(
                command_encoder,
                &self.page_table_bind_group,
                &self.volume_render_result_extent,
            );
        } else {
            self.volume_render_pass.encode(
                command_encoder,
                &self.volume_render_bind_group,
                &self.volume_render_result_extent,
            );
        }

        self.present_to_screen_pass.encode(
            command_encoder,
            &self.present_to_screen_bind_group,
            render_target,
        );
    }
}
