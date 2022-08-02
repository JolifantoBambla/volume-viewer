pub mod camera;
pub mod context;
pub mod geometry;
pub mod gpu_list;
pub mod pass;
pub mod resources;
pub mod trivial_volume_renderer;
pub mod volume;
pub mod wgsl;

use wasm_bindgen::prelude::*;

use crate::renderer::camera::Camera;
use crate::renderer::context::{ContextDescriptor, GPUContext};
use crate::renderer::pass::{dvr, present_to_screen};
use crate::renderer::pass::{ray_guided_dvr, GPUPass};

use crate::renderer::volume::RawVolumeBlock;
use crate::renderer::wgsl::create_wgsl_preprocessor;
use bytemuck;
use std::sync::Arc;
use wasm_bindgen::JsCast;
use web_sys::OffscreenCanvas;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, Buffer, Extent3d, SamplerDescriptor, SubmissionIndex};
use wgsl_preprocessor::WGSLPreprocessor;
use winit::dpi::PhysicalSize;

use crate::renderer::pass::present_to_screen::PresentToScreen;
use crate::renderer::pass::ray_guided_dvr::{RayGuidedDVR, Resources};
use crate::{SparseResidencyTexture3D, SparseResidencyTexture3DSource};
pub use trivial_volume_renderer::TrivialVolumeRenderer;

pub struct MultiChannelVolumeRenderer {
    pub(crate) ctx: Arc<GPUContext>,
    pub(crate) window_size: PhysicalSize<u32>,
    volume_transform: glam::Mat4,
    wgsl_preprocessor: WGSLPreprocessor,
    volume_texture: SparseResidencyTexture3D,

    volume_render_pass: RayGuidedDVR,
    volume_render_bind_group: BindGroup,
    volume_render_uniform_buffer: Buffer,
    volume_render_result_extent: Extent3d,

    present_to_screen_pass: PresentToScreen,
    present_to_screen_bind_group: BindGroup,
}

impl MultiChannelVolumeRenderer {
    #[cfg(target_arch = "wasm32")]
    pub async fn new(
        canvas: JsValue,
        volume_source: Box<dyn SparseResidencyTexture3DSource>,
    ) -> Self {
        let canvas = canvas.unchecked_into::<OffscreenCanvas>();
        let ctx = Arc::new(
            GPUContext::new(&ContextDescriptor::default())
                .await
                .with_surface_from_offscreen_canvas(&canvas),
        );
        let window_size = PhysicalSize {
            width: canvas.width(),
            height: canvas.height(),
        };

        let wgsl_preprocessor = create_wgsl_preprocessor();
        let volume_texture = SparseResidencyTexture3D::new(volume_source, &wgsl_preprocessor, &ctx);

        let volume_render_result_extent = Extent3d {
            width: window_size.width,
            height: window_size.height,
            depth_or_array_layers: 1,
        };

        // todo: make size configurable
        let dvr_result = resources::Texture::create_storage_texture(
            &ctx.device,
            volume_render_result_extent.width,
            volume_render_result_extent.height,
        );
        let volume_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let screen_space_sampler = ctx.device.create_sampler(&SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // the volume is a unit cube ([0,1]^3)
        // we translate it s.t. its center is the origin and scale it to its original dimensions
        let volume_transform = glam::Mat4::from_scale(volume_texture.volume_scale()).mul_mat4(
            &glam::Mat4::from_translation(glam::Vec3::new(-0.5, -0.5, -0.5)),
        );
        let uniforms = ray_guided_dvr::Uniforms::default();
        let volume_render_uniform_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let volume_render_pass = RayGuidedDVR::new(&volume_texture, &wgsl_preprocessor, &ctx);
        let volume_render_bind_group = volume_render_pass.create_bind_group(Resources {
            volume_sampler: &volume_sampler,
            output: &dvr_result.view,
            uniforms: &volume_render_uniform_buffer,
        });

        let present_to_screen_pass = PresentToScreen::new(&ctx);
        let present_to_screen_bind_group =
            present_to_screen_pass.create_bind_group(present_to_screen::Resources {
                sampler: &screen_space_sampler,
                source_texture: &dvr_result.view,
            });

        Self {
            ctx,
            window_size,
            volume_transform,
            wgsl_preprocessor,
            volume_texture,
            volume_render_pass,
            volume_render_bind_group,
            volume_render_uniform_buffer,
            volume_render_result_extent,
            present_to_screen_pass,
            present_to_screen_bind_group,
        }
    }

    pub fn update(&self, camera: &Camera, frame_number: u32) {
        let uniforms = ray_guided_dvr::Uniforms::new(
            camera.create_uniform(),
            self.volume_transform,
            frame_number,
        );
        self.ctx.queue.write_buffer(
            &self.volume_render_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    pub fn render(&self, surface_view: &wgpu::TextureView, frame_numer: u32) -> SubmissionIndex {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.volume_render_pass.encode(
            &mut encoder,
            &self.volume_render_bind_group,
            &self.volume_render_result_extent,
        );
        self.present_to_screen_pass.encode(
            &mut encoder,
            &self.present_to_screen_bind_group,
            surface_view,
        );

        // todo: process request & usage buffers
        self.volume_texture
            .encode_cache_management(&mut encoder, frame_numer);

        self.ctx.queue.submit(Some(encoder.finish()))
    }

    pub async fn post_render(&mut self, submission_index: SubmissionIndex) {
        self.volume_texture.update_cache(submission_index)
            .await;
    }
}
