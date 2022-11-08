use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::{MappableBuffer, Texture};
use crate::util::extent::extent_volume;
use crate::GPUContext;
use std::borrow::Cow;
use std::mem::size_of;
use std::process::Command;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
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

    group_wise_scan_pipeline: wgpu::ComputePipeline,
    group_wise_scan_bind_group_layout: BindGroupLayout,
    group_wise_scan_internal_bind_group: BindGroup,
    num_group_wise_scan_work_groups: u32,

    accumulate_and_update_lru_pipeline: wgpu::ComputePipeline,
    accumulate_and_update_lru_bind_group_layout: BindGroupLayout,
    accumulate_and_update_lru_internal_bind_group: BindGroup,

    scan_even: Buffer,
    scan_odd: Buffer,

    // todo: remove (debug)
    pub scan_even_read_buffer: MappableBuffer<u32>,
    pub scan_odd_read_buffer: MappableBuffer<u32>,
    pub num_entries: u32,

    state_buffer: Buffer,
    state_init: Vec<u32>,
}

impl LRUUpdate {
    pub fn new(
        num_entries: u32,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let num_workgroups = (num_entries as f32 / 128.).ceil() as u32;
        let state_init = vec![0u32; num_workgroups as usize];
        let state_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(state_init.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let scan_buffer_descriptor = BufferDescriptor {
            label: None,
            size: (size_of::<u32>() * num_entries as usize) as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, // todo: remove copy_src
            mapped_at_creation: false,
        };
        // todo: remove (debug)
        let scan_buffer_read_descriptor = BufferDescriptor {
            label: None,
            size: (size_of::<u32>() * num_entries as usize) as BufferAddress,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let scan_even = ctx.device.create_buffer(&scan_buffer_descriptor);
        let scan_odd = ctx.device.create_buffer(&scan_buffer_descriptor);

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
        let group_wise_scan_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader_module,
                    entry_point: "group_wise_scan",
                });

        let group_wise_scan_bind_group_layout = group_wise_scan_pipeline.get_bind_group_layout(0);

        let group_wise_scan_internal_bind_group_layout =
            group_wise_scan_pipeline.get_bind_group_layout(1);
        let group_wise_scan_internal_bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &group_wise_scan_internal_bind_group_layout,
                entries: &vec![
                    BindGroupEntry {
                        binding: 0,
                        resource: scan_even.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: scan_odd.as_entire_binding(),
                    },
                ],
            });

        let accumulate_and_update_lru_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader_module,
                    entry_point: "accumulate_and_update_lru",
                });
        let accumulate_and_update_lru_bind_group_layout =
            accumulate_and_update_lru_pipeline.get_bind_group_layout(0);

        let accumulate_and_update_lru_internal_bind_group_layout =
            accumulate_and_update_lru_pipeline.get_bind_group_layout(1);
        let accumulate_and_update_lru_internal_bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &accumulate_and_update_lru_internal_bind_group_layout,
                entries: &vec![
                    BindGroupEntry {
                        binding: 0,
                        resource: scan_even.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: scan_odd.as_entire_binding(),
                    },
                ],
            });

        Self {
            ctx: ctx.clone(),

            group_wise_scan_pipeline,
            group_wise_scan_bind_group_layout,
            group_wise_scan_internal_bind_group,
            num_group_wise_scan_work_groups: num_workgroups,

            accumulate_and_update_lru_pipeline,
            accumulate_and_update_lru_bind_group_layout,
            accumulate_and_update_lru_internal_bind_group,

            scan_even,
            scan_odd,

            scan_even_read_buffer: MappableBuffer::new(
                ctx.device.create_buffer(&scan_buffer_read_descriptor),
            ),
            scan_odd_read_buffer: MappableBuffer::new(
                ctx.device.create_buffer(&scan_buffer_read_descriptor),
            ),
            num_entries,

            state_buffer,
            state_init,
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut CommandEncoder,
        bind_groups: &Vec<BindGroup>,
        lru_buffer: &Buffer,
        output_extent: &wgpu::Extent3d,
    ) {
        self.ctx.queue.write_buffer(
            &self.state_buffer,
            0,
            bytemuck::cast_slice(self.state_init.as_slice()),
        );

        self.encode_group_wise_scan(command_encoder, &bind_groups[0], output_extent);
        self.encode_accumulate_and_update_lru(command_encoder, &bind_groups[1]);
        self.encode_copy_lru(command_encoder, lru_buffer);
    }

    fn encode_copy_lru(&self, command_encoder: &mut CommandEncoder, lru_buffer: &Buffer) {
        let buffer_size = (size_of::<u32>() as u32 * self.num_entries) as BufferAddress;
        command_encoder.copy_buffer_to_buffer(&self.scan_odd, 0, lru_buffer, 0, buffer_size);
    }

    fn encode_group_wise_scan(
        &self,
        command_encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        output_extent: &wgpu::Extent3d,
    ) {
        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LRU Update"),
        });
        cpass.set_pipeline(&self.group_wise_scan_pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.set_bind_group(1, &self.group_wise_scan_internal_bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch_workgroups(
            (extent_volume(output_extent) as f32 / 128.).ceil() as u32,
            1,
            1,
        );
    }

    fn encode_accumulate_and_update_lru(
        &self,
        command_encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
    ) {
        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LRU Update"),
        });
        cpass.set_pipeline(&self.accumulate_and_update_lru_pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.set_bind_group(1, &self.accumulate_and_update_lru_internal_bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // todo: remove(debug)
    pub fn encode_copy(&self, command_encoder: &mut CommandEncoder) {
        if self.scan_even_read_buffer.is_ready() && self.scan_odd_read_buffer.is_ready() {
            let buffer_size = (size_of::<u32>() as u32 * self.num_entries) as BufferAddress;
            command_encoder.copy_buffer_to_buffer(
                &self.scan_even,
                0,
                self.scan_even_read_buffer.as_buffer_ref(),
                0,
                buffer_size,
            );
            command_encoder.copy_buffer_to_buffer(
                &self.scan_odd,
                0,
                self.scan_odd_read_buffer.as_buffer_ref(),
                0,
                buffer_size,
            );
        }
    }

    pub fn create_bind_groups(&self, resources: &Resources) -> Vec<BindGroup> {
        vec![
            self.ctx()
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.group_wise_scan_bind_group_layout,
                    entries: &resources.as_bind_group_entries(),
                }),
            self.ctx()
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.accumulate_and_update_lru_bind_group_layout,
                    entries: &resources.as_bind_group_entries(),
                }),
        ]
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
        &self.group_wise_scan_bind_group_layout
    }
}
