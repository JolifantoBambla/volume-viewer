use crate::renderer::pass::scan::Scan;
use crate::renderer::pass::{ComputePipelineData, StaticComputeEncodeDescriptor};
use crate::resource::buffer::TypedBuffer;
use crate::resource::sparse_residency::texture3d::cache_management::lru::NumUsedEntries;
use crate::resource::Texture;
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferUsages, CommandEncoder,
    ComputePassDescriptor, ComputePipelineDescriptor, Device, Label,
};
use wgpu_framework::context::Gpu;
use wgsl_preprocessor::WGSLPreprocessor;

const WORKGROUP_SIZE: u32 = 256;

pub struct LRUUpdateResources<'a> {
    lru_cache: Arc<TypedBuffer<u32>>,
    num_used_entries: Arc<TypedBuffer<NumUsedEntries>>,
    usage_buffer: &'a Texture,
    // todo: make that into a TypedBuffer
    time_stamp: &'a Buffer,
}

impl<'a> LRUUpdateResources<'a> {
    pub fn new(
        lru_cache: Arc<TypedBuffer<u32>>,
        num_used_entries: Arc<TypedBuffer<NumUsedEntries>>,
        usage_buffer: &'a Texture,
        time_stamp: &'a Buffer,
    ) -> Self {
        Self {
            lru_cache,
            num_used_entries,
            usage_buffer,
            time_stamp,
        }
    }
}

#[derive(Debug)]
pub(crate) struct LRUUpdate {
    ctx: Arc<Gpu>,

    // these two resources get updated
    lru_cache: Arc<TypedBuffer<u32>>,
    num_used_entries: Arc<TypedBuffer<NumUsedEntries>>,

    // temp buffer
    lru_updated: TypedBuffer<u32>,

    initialize_offsets_pass: StaticComputeEncodeDescriptor,
    scan: Scan,
    update_lru_pass: StaticComputeEncodeDescriptor,
}

impl LRUUpdate {
    fn create_base_bind_group<'a, const N: usize>(
        label: &str,
        pipeline: &ComputePipelineData<N>,
        resources: &'a LRUUpdateResources<'a>,
        offsets: &TypedBuffer<u32>,
        used: &TypedBuffer<u32>,
        device: &Device,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Label::from(label),
            layout: pipeline.bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: resources.lru_cache.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: resources.num_used_entries.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: offsets.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: used.buffer().as_entire_binding(),
                },
            ],
        })
    }

    pub fn new<'a>(
        resources: &'a LRUUpdateResources<'a>,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<Gpu>,
    ) -> Self {
        let mut preprocessor = wgsl_preprocessor.clone();
        preprocessor.include("lru_update_base", include_str!("lru_update_base.wgsl"));

        let init_shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &preprocessor
                        .preprocess(include_str!("lru_update_init.wgsl"))
                        .ok()
                        .unwrap(),
                )),
            });

        let sort_shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &preprocessor
                        .preprocess(include_str!("lru_update_sort.wgsl"))
                        .ok()
                        .unwrap(),
                )),
            });

        let lru_update_init_pipeline: Rc<ComputePipelineData<2>> =
            Rc::new(ComputePipelineData::new(
                &ComputePipelineDescriptor {
                    label: Label::from("initialize offsets"),
                    layout: None,
                    module: &init_shader,
                    entry_point: "main",
                },
                ctx.device(),
            ));
        let lru_update_sort_pipeline: Rc<ComputePipelineData<2>> =
            Rc::new(ComputePipelineData::new(
                &ComputePipelineDescriptor {
                    label: Label::from("update lru"),
                    layout: None,
                    module: &sort_shader,
                    entry_point: "main",
                },
                ctx.device(),
            ));

        let offsets: TypedBuffer<u32> = TypedBuffer::new_zeroed(
            "offsets",
            resources.lru_cache.num_elements(),
            BufferUsages::STORAGE,
            ctx.device(),
        );
        let used: TypedBuffer<u32> = TypedBuffer::new_zeroed(
            "used",
            resources.lru_cache.num_elements(),
            BufferUsages::STORAGE,
            ctx.device(),
        );

        let lru_updated: TypedBuffer<u32> = TypedBuffer::new_zeroed(
            "lru updated",
            resources.lru_cache.num_elements(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ctx.device(),
        );

        let scan = Scan::new(&offsets, wgsl_preprocessor, ctx);

        let initialize_offsets_pass = StaticComputeEncodeDescriptor::new_1d(
            lru_update_init_pipeline.pipeline(),
            vec![
                LRUUpdate::create_base_bind_group(
                    "LRU update init 0",
                    &lru_update_init_pipeline,
                    resources,
                    &offsets,
                    &used,
                    ctx.device(),
                ),
                ctx.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("LRU update init 1"),
                    layout: lru_update_init_pipeline.bind_group_layout(1),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &resources.usage_buffer.view,
                            ),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: resources.time_stamp.as_entire_binding(),
                        },
                    ],
                }),
            ],
            WORKGROUP_SIZE,
        );
        let update_lru_pass = StaticComputeEncodeDescriptor::new_1d(
            lru_update_sort_pipeline.pipeline(),
            vec![
                LRUUpdate::create_base_bind_group(
                    "LRU update sort 0",
                    &lru_update_sort_pipeline,
                    resources,
                    &offsets,
                    &used,
                    ctx.device(),
                ),
                ctx.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("LRU update sort 1"),
                    layout: lru_update_sort_pipeline.bind_group_layout(1),
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: lru_updated.buffer().as_entire_binding(),
                    }],
                }),
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

    pub fn encode(
        &self,
        command_encoder: &mut CommandEncoder
    ) {
        self.ctx.queue().write_buffer(
            self.num_used_entries.buffer(),
            0,
            bytemuck::bytes_of(&NumUsedEntries { num: 0 }),
        );
        self.encode_passes(
            command_encoder
        );
        self.encode_copy(command_encoder);
    }

    fn encode_passes(
        &self,
        command_encoder: &mut CommandEncoder
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Label::from("update lru"),
            timestamp_writes: None,
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
            self.lru_cache.size(),
        );
    }
}
