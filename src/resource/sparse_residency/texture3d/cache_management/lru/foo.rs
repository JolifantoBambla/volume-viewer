use crate::renderer::pass::{AsBindGroupEntries, ComputeEncodeDescriptor, ComputePipelineData, GPUPass};
use crate::resource::Texture;
use crate::util::extent::extent_volume;
use crate::GPUContext;
use std::borrow::Cow;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass, ComputePassDescriptor, ComputePipelineDescriptor, Device, Label};
use wgsl_preprocessor::WGSLPreprocessor;
use crate::renderer::pass::scan::Scan;
use crate::resource::buffer::TypedBuffer;
use crate::resource::sparse_residency::texture3d::cache_management::lru::NumUsedEntries;

const WORKGROUP_SIZE: u32 = 256;

pub struct LRUUpdateResources<'a> {
    lru_cache: Rc<TypedBuffer<u32>>,
    num_used_entries: Rc<TypedBuffer<NumUsedEntries>>,
    usage_buffer: &'a Texture,
    // todo: make that into a TypedBuffer
    time_stamp: &'a Buffer,
}

pub(crate) struct LRUUpdate {
    ctx: Arc<GPUContext>,

    // these two resources get updated
    lru_cache: Rc<TypedBuffer<u32>>,
    num_used_entries: Rc<TypedBuffer<NumUsedEntries>>,

    // temp buffer
    lru_updated: TypedBuffer<u32>,

    initialize_offsets_pass: ComputeEncodeDescriptor,
    scan: Scan,
    update_lru_pass: ComputeEncodeDescriptor,
}

impl LRUUpdate {
    fn create_base_bind_group<'a, const N: usize>(
        label: &str,
        pipeline: &ComputePipelineData<N>,
        resources: &'a LRUUpdateResources<'a>,
        offsets: &TypedBuffer<u32>,
        device: &Device,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Label::from(label),
            layout: pipeline.bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&resources.usage_buffer.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: resources.time_stamp.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: resources.lru_cache.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: resources.num_used_entries.buffer().as_entire_binding()
                },
                BindGroupEntry {
                    binding: 4,
                    resource: offsets.buffer().as_entire_binding(),
                }
            ]
        })
    }

    pub fn new<'a>(
        resources: &'a LRUUpdateResources<'a>,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        // todo: create internal resources
        //  - offsets
        //  - lru_updated
        // todo: create pipeline for initializing offsets
        // todo: create Scan for offsets
        // todo: create pipeline for updating lru
        // todo: don't forget to copy updated lru to lru from resources

        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("foo.wgsl"))
                        .ok()
                        .unwrap(),
                )),
            });
        let initialize_offsets_pipeline: Rc<ComputePipelineData<1>> = Rc::new(
            ComputePipelineData::new(
                &ComputePipelineDescriptor {
                    label: Label::from("initialize offsets"),
                    layout: None,
                    module: &shader_module,
                    entry_point: "initialize_offsets",
                },
                &ctx.device
            )
        );
        let update_lru_pipeline: Rc<ComputePipelineData<2>> = Rc::new(
            ComputePipelineData::new(
                &ComputePipelineDescriptor {
                    label: Label::from("update lru"),
                    layout: None,
                    module: &shader_module,
                    entry_point: "update_lru",
                },
                &ctx.device
            )
        );

        let offsets: TypedBuffer<u32> = TypedBuffer::new_zeroed(
            "offsets",
            resources.lru_cache.num_elements(),
            BufferUsages::STORAGE,
            &ctx.device,
        );
        let lru_updated: TypedBuffer<u32> = TypedBuffer::new_zeroed(
            "lru updated",
            resources.lru_cache.num_elements(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            &ctx.device,
        );

        let scan = Scan::new(&offsets, wgsl_preprocessor, ctx);

        let initialize_offsets_pass = ComputeEncodeDescriptor::new_1d(
            initialize_offsets_pipeline.pipeline(),
            vec![
                LRUUpdate::create_base_bind_group(
                    "initialize offsets",
                    &initialize_offsets_pipeline,
                    resources,
                    &offsets,
                    &ctx.device
                )
            ],
            WORKGROUP_SIZE,
        );
        let update_lru_pass = ComputeEncodeDescriptor::new_1d(
            update_lru_pipeline.pipeline(),
            vec![
                LRUUpdate::create_base_bind_group(
                    "update lru 0",
                    &update_lru_pipeline,
                    resources,
                    &offsets,
                    &ctx.device
                ),
                ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: Label::from("update lru 1"),
                    layout: update_lru_pipeline.bind_group_layout(1),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: lru_updated.buffer().as_entire_binding(),
                        }
                    ]
                })
            ],
            WORKGROUP_SIZE,
        );

        Self {
            ctx: ctx.clone(),
            lru_cache: resources.lru_cache.clone(),
            num_used_entries: resources.num_used_entries.clone(),
            lru_updated,
            initialize_offsets_pass,
            scan,
            update_lru_pass,
        }
    }

    pub fn encode(&self, command_encoder: &mut CommandEncoder) {
        self.ctx.queue.write_buffer(
            self.num_used_entries.buffer(),
            0,
            bytemuck::bytes_of(&NumUsedEntries { num: 0 }),
        );
        self.encode_passes(command_encoder);
        self.encode_copy(command_encoder);
    }

    fn encode_passes(&self, command_encoder: &mut CommandEncoder) {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Label::from("update lru")
        });
        self.initialize_offsets_pass.encode(&mut compute_pass);
        self.scan.encode_to_pass(&mut compute_pass);
        self.update_lru_pass.encode(&mut compute_pass);
    }

    fn encode_copy(&self, command_encoder: &mut CommandEncoder) {
        command_encoder.copy_buffer_to_buffer(
            self.lru_updated.buffer(),
            0,
            self.lru_cache.buffer(),
            0,
            self.lru_cache.size()
        );
    }
}
