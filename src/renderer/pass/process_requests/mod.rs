use crate::renderer::camera::TransformUniform;
use crate::renderer::gpu_list::GpuList;
use crate::renderer::resources::Texture;
use crate::renderer::{
    camera::CameraUniform,
    context::GPUContext,
    pass::{AsBindGroupEntries, GPUPass},
};
use crate::SparseResidencyTexture3D;
use glam::UVec4;
use std::{borrow::Cow, sync::Arc};
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, Buffer, CommandEncoder};
use wgsl_preprocessor::WGSLPreprocessor;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Timestamp {
    pub now: u32,
}

impl Timestamp {
    pub fn new(now: u32) -> Self {
        Self { now }
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self { now: 0 }
    }
}

pub struct Resources<'a> {
    pub page_table_meta: &'a Buffer,
    pub request_buffer: &'a Texture,
    pub timestamp: &'a Buffer,
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
                resource: self.timestamp.as_entire_binding(),
            },
        ]
    }
}

pub struct ProcessRequests {
    ctx: Arc<GPUContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: BindGroupLayout,
    internal_bind_group: BindGroup,
    request_list: GpuList<u32>,
}

impl ProcessRequests {
    pub fn new(
        max_requests: u32,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let request_list = GpuList::new(None, max_requests, &ctx);
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
        let internal_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &internal_bind_group_layout,
            entries: &request_list.as_bind_group_entries(),
        });

        Self {
            ctx: ctx.clone(),
            pipeline,
            bind_group_layout,
            internal_bind_group,
            request_list,
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        bind_group: &BindGroup,
        output_extent: &wgpu::Extent3d,
    ) {
        self.request_list.clear();

        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Process Requests"),
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.set_bind_group(1, &self.internal_bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch_workgroups(
            (output_extent.width as f32 / 4.).ceil() as u32,
            (output_extent.height as f32 / 4.).ceil() as u32,
            (output_extent.depth_or_array_layers as f32 / 4.).ceil() as u32,
        );
    }

    pub fn encode_copy_result_to_readable(&self, encoder: &mut CommandEncoder) {
        self.request_list.copy_to_readable(encoder);
    }

    pub fn map_for_reading(&self) {
        self.request_list.map_for_reading();
    }

    pub fn read(&self) -> Vec<u32> {
        self.request_list.read_mapped()
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
