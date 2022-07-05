use crate::renderer::{
    context::GPUContext,
    pass::{AsBindGroupEntries, GPUPass},
};
use std::{borrow::Cow, sync::Arc};
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub slice: i32,
    pub max: f32,
}

pub struct Resources<'a> {
    pub volume: &'a wgpu::TextureView,
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
                resource: wgpu::BindingResource::TextureView(self.output),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.uniforms.as_entire_binding(),
            },
        ]
    }
}

pub struct NormalizeZSlice {
    ctx: Arc<GPUContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl NormalizeZSlice {
    pub fn new(ctx: &Arc<GPUContext>) -> Self {
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "normalize_z_slice.wgsl"
                ))),
            });
        let pipeline = ctx
            .device
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
        volume_extent: &wgpu::Extent3d,
    ) {
        let mut cpass =
            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch_workgroups(volume_extent.width / 16, volume_extent.height / 16, 1);
    }
}

impl<'a> GPUPass<Resources<'a>> for NormalizeZSlice {
    fn ctx(&self) -> &Arc<GPUContext> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "NormalizeZSlice"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
