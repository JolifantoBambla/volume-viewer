use crate::renderer::{
    camera::CameraUniform,
    context::GPUContext,
    pass::{AsBindGroupEntries, GPUPass},
};
use std::{borrow::Cow, sync::Arc};
use glam::UVec4;
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, Buffer};
use wgsl_preprocessor::WGSLPreprocessor;
use crate::renderer::camera::TransformUniform;
use crate::renderer::resources::Texture;
use crate::SparseResidencyTexture3D;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub max_requests: u32,
    pub timestamp: u32,
}

impl Params {
    pub fn new(max_requests: u32, timestamp: u32) -> Self {
        Self {
            max_requests,
            timestamp,
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_requests: 0,
            timestamp: 0,
        }
    }
}

pub struct Resources<'a> {
    pub page_table_meta: &'a Buffer,
    pub request_buffer: &'a Texture,
    pub params: &'a Buffer,
    pub counters: &'a Buffer,
    pub ids: &'a Buffer,
}

impl<'a> AsBindGroupEntries for Resources<'a> {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.page_table_meta.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&self.request_buffer.view),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.page_table_meta.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.page_table_meta.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: self.page_table_meta.as_entire_binding(),
            },
        ]
    }
}

pub struct ProcessRequests {
    ctx: Arc<GPUContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: BindGroupLayout,
    internal_bind_group: BindGroup,
}

impl ProcessRequests {
    pub fn new(volume_texture: &SparseResidencyTexture3D, wgsl_preprocessor: &WGSLPreprocessor, ctx: &Arc<GPUContext>) -> Self {
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("process_requests.wgsl"))
                        .ok()
                        .unwrap(),
                )),
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

        let internal_bind_group_layout = pipeline.get_bind_group_layout(1);
        let internal_bind_group = ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &internal_bind_group_layout,
                entries: &volume_texture.as_bind_group_entries(),
            });

        Self {
            ctx: ctx.clone(),
            pipeline,
            bind_group_layout,
            internal_bind_group,
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        max_requests: u32,
        timestamp: u32,
    ) {
        // todo: write zeros to buffers to buffers

        let mut cpass =
            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray Guided DVR")
            });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.set_bind_group(1, &self.internal_bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch_workgroups(
            (output_extent.width as f32 / 16.).ceil() as u32,
            (output_extent.height as f32 / 16.).ceil() as u32,
            1,
        );
    }
}

impl<'a> GPUPass<Resources<'a>> for ProcessRequests {
    fn ctx(&self) -> &Arc<GPUContext> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "Process Requests"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
