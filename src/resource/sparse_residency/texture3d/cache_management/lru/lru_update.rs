use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::Texture;
use crate::GPUContext;
use std::borrow::Cow;
use std::mem::size_of;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, Buffer, BufferAddress, BufferDescriptor,
    BufferUsages, CommandEncoder,
};
use wgsl_preprocessor::WGSLPreprocessor;

pub struct Resources<'a> {
    pub usage_buffer: &'a Texture,
    pub timestamp: &'a Buffer,
    pub lru_cache: &'a Buffer,
    pub num_used_entries: &'a Buffer,
}

impl<'a> AsBindGroupEntries for Resources<'a> {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.usage_buffer.view),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.timestamp.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.lru_cache.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.num_used_entries.as_entire_binding(),
            },
        ]
    }
}

pub(crate) struct LRUUpdate {
    ctx: Arc<GPUContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: BindGroupLayout,
    internal_bind_group: BindGroup,
    usage_mask: Buffer,
    scan_even: Buffer,
    scan_odd: Buffer,
}

impl LRUUpdate {
    pub fn new(
        num_entries: u32,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let usage_mask = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (size_of::<u32>() * num_entries as usize) as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scan_even = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (size_of::<u32>() * num_entries as usize) as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scan_odd = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (size_of::<u32>() * num_entries as usize) as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("lru_update.wgsl"))
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
            entries: &vec![
                BindGroupEntry {
                    binding: 0,
                    resource: usage_mask.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: scan_even.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: scan_odd.as_entire_binding(),
                },
            ],
        });

        Self {
            ctx: ctx.clone(),
            pipeline,
            bind_group_layout,
            internal_bind_group,
            usage_mask,
            scan_even,
            scan_odd,
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        output_extent: &wgpu::Extent3d,
    ) {
        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Create Mask"),
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
}

impl<'a> GPUPass<Resources<'a>> for LRUUpdate {
    fn ctx(&self) -> &Arc<GPUContext> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "LRU Update"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
