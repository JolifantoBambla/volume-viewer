use std::{
    borrow::Cow,
    sync::Arc,
};
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout};
use crate::renderer::{
    context::GPUContext,
    pass::{GPUPass, AsBindGroupEntries},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub world_to_object: glam::Mat4,
    pub screen_to_camera: glam::Mat4,
    pub camera_to_world: glam::Mat4,
    pub volume_color: glam::Vec4,
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            world_to_object: glam::Mat4::IDENTITY,
            screen_to_camera: glam::Mat4::IDENTITY,
            camera_to_world: glam::Mat4::IDENTITY,
            volume_color: glam::Vec4::new(0.,0.,1., 1.),
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(self.volume),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(self.volume_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(self.output),
            },
            wgpu::BindGroupEntry{
                binding: 3,
                resource: self.uniforms.as_entire_binding(),
            },
        ]
    }
}

pub struct DVR {
    ctx: Arc<GPUContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl DVR {
    pub fn new(ctx: &Arc<GPUContext>) -> Self {
        let shader_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("dvr.wgsl"))),
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

    pub fn encode(&self, command_encoder: &mut wgpu::CommandEncoder, bind_group: &BindGroup, output_extent: &wgpu::Extent3d) {
        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch((output_extent.width as f32/ 16.).ceil() as u32, (output_extent.height as f32 / 16.).ceil() as u32, 1);
    }
}

impl<'a> GPUPass<Resources<'a>> for DVR {
    fn ctx(&self) -> &Arc<GPUContext> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "DVR"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
