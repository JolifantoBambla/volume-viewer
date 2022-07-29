use bytemuck;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::OffscreenCanvas;
use wgpu::util::DeviceExt;
use wgsl_preprocessor::WGSLPreprocessor;
use crate::{Camera, RawVolumeBlock};
use crate::renderer::context::{ContextDescriptor, GPUContext};
use crate::renderer::pass::GPUPass;
use crate::renderer::pass::{dvr, present_to_screen};
use crate::renderer::resources;
use crate::renderer::wgsl::create_wgsl_preprocessor;

pub struct TrivialVolumeRenderer {
    pub(crate) canvas: OffscreenCanvas,
    pub(crate) ctx: Arc<GPUContext>,

    dvr_pass: dvr::DVR,
    present_to_screen: present_to_screen::PresentToScreen,

    dvr_bind_group: wgpu::BindGroup,
    present_to_screen_bind_group: wgpu::BindGroup,

    dvr_result_extent: wgpu::Extent3d,

    volume_transform: glam::Mat4,
    uniform_buffer: wgpu::Buffer,

    wgsl_preprocessor: WGSLPreprocessor,
}

impl TrivialVolumeRenderer {
    pub async fn new(canvas: JsValue, volume: RawVolumeBlock) -> Self {
        let canvas = canvas.unchecked_into::<OffscreenCanvas>();
        let ctx = Arc::new(
            GPUContext::new(&ContextDescriptor::default())
                .await
                .with_surface_from_offscreen_canvas(&canvas),
        );

        let wgsl_preprocessor = create_wgsl_preprocessor();

        let volume_texture =
            resources::Texture::from_raw_volume_block(&ctx.device, &ctx.queue, &volume);
        let storage_texture = resources::Texture::create_storage_texture(
            &ctx.device,
            canvas.width(),
            canvas.height(),
        );

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // the volume is a unit cube ([0,1]^3)
        // we translate it s.t. its center is the origin and scale it to its original dimensions
        // todo: scale should come from volume meta data (-> todo: add meta data to volume)
        let volume_transform = glam::Mat4::from_scale(volume.create_vec3()).mul_mat4(
            &glam::Mat4::from_translation(glam::Vec3::new(-0.5, -0.5, -0.5)),
        );

        let uniforms = dvr::Uniforms {
            world_to_object: volume_transform,
            ..Default::default()
        };
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let dvr_pass = dvr::DVR::new(&wgsl_preprocessor, &ctx);
        let full_screen_pass = present_to_screen::PresentToScreen::new(&ctx);
        let dvr_bind_group = dvr_pass.create_bind_group(dvr::Resources {
            volume: &volume_texture.view,
            volume_sampler: &sampler,
            output: &storage_texture.view,
            uniforms: &uniform_buffer,
        });
        let full_screen_bind_group =
            full_screen_pass.create_bind_group(present_to_screen::Resources {
                sampler: &sampler,
                source_texture: &storage_texture.view,
            });

        let dvr_result_extent = wgpu::Extent3d {
            width: canvas.width(),
            height: canvas.height(),
            depth_or_array_layers: 1,
        };

        Self {
            canvas,
            ctx,
            dvr_pass,
            dvr_bind_group,
            present_to_screen: full_screen_pass,
            present_to_screen_bind_group: full_screen_bind_group,
            dvr_result_extent,
            volume_transform,
            uniform_buffer,
            wgsl_preprocessor,
        }
    }

    pub fn update(&self, camera: &Camera) {
        let uniforms = dvr::Uniforms::new(camera.create_uniform(), self.volume_transform);
        self.ctx
            .queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn render(&self, canvas_view: &wgpu::TextureView) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.dvr_pass
            .encode(&mut encoder, &self.dvr_bind_group, &self.dvr_result_extent);
        self.present_to_screen.encode(
            &mut encoder,
            &self.present_to_screen_bind_group,
            canvas_view,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
    }
}
