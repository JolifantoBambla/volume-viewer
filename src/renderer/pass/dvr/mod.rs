use crate::app::renderer::common::CameraUniform;
use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use std::{borrow::Cow, sync::Arc};
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout};
use wgpu_framework::context::Gpu;
use wgsl_preprocessor::WGSLPreprocessor;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub camera: CameraUniform,
    pub world_to_object: glam::Mat4,
    pub object_to_world: glam::Mat4,
    pub volume_color: glam::Vec4,
}

impl Uniforms {
    pub fn new(camera: CameraUniform, object_to_world: glam::Mat4) -> Self {
        Self {
            camera,
            world_to_object: object_to_world.inverse(),
            object_to_world,
            volume_color: glam::Vec4::new(0., 0., 1., 1.),
        }
    }
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            camera: CameraUniform::default(),
            world_to_object: glam::Mat4::IDENTITY,
            object_to_world: glam::Mat4::IDENTITY,
            volume_color: glam::Vec4::new(0., 0., 1., 1.),
        }
    }
}

pub struct Resources<'a> {
    pub volume: &'a wgpu::TextureView,
    pub volume_sampler: &'a wgpu::Sampler,
    pub output: &'a wgpu::TextureView,
    pub uniforms: &'a wgpu::Buffer,
}

impl<'a> AsBindGroupEntries for Resources<'a> {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(self.volume),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(self.volume_sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(self.output),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.uniforms.as_entire_binding(),
            },
        ]
    }
}

pub struct DVR {
    ctx: Arc<Gpu>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl DVR {
    pub fn new(wgsl_preprocessor: &WGSLPreprocessor, ctx: &Arc<Gpu>) -> Self {
        let shader_module = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("dvr.wgsl"))
                        .ok()
                        .unwrap(),
                )),
            });
        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Self {
            ctx: ctx.clone(),
            pipeline,
            bind_group_layout,
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        bind_group: &BindGroup,
        output_extent: &wgpu::Extent3d,
    ) {
        let mut cpass =
            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                ..Default::default()
            });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch_workgroups(
            (output_extent.width as f32 / 16.).ceil() as u32,
            (output_extent.height as f32 / 16.).ceil() as u32,
            1,
        );
    }
}

impl<'a> GPUPass<Resources<'a>> for DVR {
    fn ctx(&self) -> &Arc<Gpu> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "DVR"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
