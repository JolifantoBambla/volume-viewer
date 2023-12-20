use crate::renderer::pass::{
    ComputeEncodeIndirectDescriptor, ComputePassData, ComputePipelineData,
    DispatchWorkgroupsIndirect, DynamicComputeEncodeDescriptor, StaticComputeEncodeDescriptor,
};
use crate::resource::sparse_residency::texture3d::brick_cache_update::CacheUpdateMeta;
use crate::resource::VolumeManager;
use crate::volume::octree::octree_manager::Octree;
use glam::UVec3;
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoder, ComputePass,
    ComputePassDescriptor, ComputePipelineDescriptor, Label, MapMode,
};
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::{Buffer, MappableBuffer};
use wgsl_preprocessor::WGSLPreprocessor;

const WORKGROUP_SIZE: u32 = 256;
const MIN_MAX_THREAD_BLOCK_SIZE: UVec3 = UVec3::new(2, 2, 2);

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CacheUpdateMetaGPU {
    num_mapped: u32,
    num_unmapped: u32,
    num_mapped_first_time: u32,
    num_unsuccessful_map_attempt: u32,
}

#[derive(Debug)]
struct OnFirstTimeMappedPasses {
    processing_size: u32,
    compute_min_max_pass: DynamicComputeEncodeDescriptor,
    update_min_max_values_pass: StaticComputeEncodeDescriptor,
}

impl OnFirstTimeMappedPasses {
    pub fn encode<'a>(&'a self, compute_pass: &mut ComputePass<'a>, num_new_bricks: u32) {
        if num_new_bricks > 0 {
            /*
            log::info!(
                "first time workgroups {} = new bricks {} * processing size {} / 64",
                f32::ceil(
                    num_new_bricks as f32 * self.processing_size as f32 / WORKGROUP_SIZE as f32
                ) as u32,
                num_new_bricks,
                self.processing_size
            );
             */
            self.compute_min_max_pass.encode_1d(
                compute_pass,
                f32::ceil(
                    num_new_bricks as f32 * self.processing_size as f32 / WORKGROUP_SIZE as f32,
                ) as u32,
            );
            self.update_min_max_values_pass.encode(compute_pass);
        }
    }
}

#[derive(Debug)]
struct OnMappedPasses {
    process_mapped_bricks_pass: DynamicComputeEncodeDescriptor,
    dependent_passes: Vec<ComputePassData>,
}

impl OnMappedPasses {
    pub fn encode<'a>(&'a self, compute_pass: &mut ComputePass<'a>, num_mapped_bricks: u32) {
        if num_mapped_bricks > 0 {
            /*
            log::info!(
                "mapped workgroups {} = num mapped {} / 64",
                f32::ceil(num_mapped_bricks as f32 / WORKGROUP_SIZE as f32) as u32,
                num_mapped_bricks
            );
             */
            self.process_mapped_bricks_pass.encode_1d(
                compute_pass,
                f32::ceil(num_mapped_bricks as f32 / WORKGROUP_SIZE as f32) as u32,
            );

            for pass in self.dependent_passes.iter() {
                pass.encode(compute_pass);
            }

            /*
            // todo: remove (debug)
            let mut num_processed = 0;
            let limit = 3;
            for pass in self.dependent_passes.iter() {
                if num_processed < limit {
                    pass.encode(compute_pass);
                }
                num_processed += 1;
            }
            */
        }
    }
}

#[derive(Debug)]
struct BreakPointBuffers {
    helper_buffer_a: MappableBuffer<u32>,
    helper_buffer_b: MappableBuffer<u32>,
}

impl BreakPointBuffers {
    pub fn new(helper_buffer_a: &Buffer<u32>, helper_buffer_b: &Buffer<u32>) -> Self {
        Self {
            helper_buffer_a: MappableBuffer::from_buffer(helper_buffer_a),
            helper_buffer_b: MappableBuffer::from_buffer(helper_buffer_b),
        }
    }

    pub fn copy_to_read_buffer(
        &self,
        command_encoder: &mut CommandEncoder,
        helper_buffer_a: &Buffer<u32>,
        helper_buffer_b: &Buffer<u32>,
    ) {
        if self.helper_buffer_a.is_ready() {
            //log::info!("copy to a");
            command_encoder.copy_buffer_to_buffer(
                helper_buffer_a.buffer(),
                0,
                self.helper_buffer_a.buffer(),
                0,
                helper_buffer_a.size(),
            );
        }
        if self.helper_buffer_b.is_ready() {
            //log::info!("copy to b");
            command_encoder.copy_buffer_to_buffer(
                helper_buffer_b.buffer(),
                0,
                self.helper_buffer_b.buffer(),
                0,
                helper_buffer_b.size(),
            );
        }
    }

    pub fn map(&self) {
        if self.helper_buffer_a.map_async(MapMode::Read, ..).is_err() {
            log::info!("could not map buffer a");
        }
        if self.helper_buffer_b.map_async(MapMode::Read, ..).is_err() {
            log::info!("could not map buffer b")
        }
    }

    pub fn maybe_print(&self) -> bool {
        if self.helper_buffer_a.is_mapped() && self.helper_buffer_b.is_mapped() {
            log::info!(
                "helper buffer a {:?}",
                self.helper_buffer_a.read_all().unwrap()
            );
            log::info!(
                "helper buffer b {:?}",
                self.helper_buffer_b.read_all().unwrap()
            );
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct OctreeUpdate {
    #[allow(unused)]
    gpu: Arc<Gpu>,

    // buffers that need updating every frame
    cache_update_meta_buffer: Buffer<CacheUpdateMetaGPU>,
    new_brick_ids_buffer: Buffer<u32>,
    mapped_brick_ids_buffer: Buffer<u32>,
    unmapped_brick_ids_buffer: Buffer<u32>,
    indirect_buffers: Vec<Rc<Buffer<DispatchWorkgroupsIndirect>>>,
    num_nodes_to_update_buffers: Vec<Buffer<u32>>,

    // buffers that must not go out of scope
    #[allow(unused)]
    helper_buffer_a: Buffer<u32>,
    #[allow(unused)]
    helper_buffer_b: Buffer<u32>,
    #[allow(unused)]
    subdivision_index_buffers: Vec<Buffer<u32>>,

    // passes
    on_mapped_first_time_passes: OnFirstTimeMappedPasses,
    on_mapped_passes: OnMappedPasses,

    // todo: remove (debug)
    break_point: Option<BreakPointBuffers>,
}

// with min max:
// 1. compute min max:
//  - compute min and store in helper a (atomicMin)
//  - compute max and store in helper b (atomicMax)
// 2. update min max:
//  - if min or max in helper buffers is not default value
//  - update node
//  - mark in helper buffer b if min max changed
//  - set helper buffer a to 255
// 2. process mapped:
//  - compute bitmask for resolution
//  - read node and check if bitmask is different
//  - if so, atomicOr in helper buffer b
// 3. process helper buffers:
//  - if helper buffer a non-zero: update node & collect parent node in helper buffer b
//  - set helper buffer a to 255
// 5. process parent node buffer:
//  - if helper buffer b non-zero: atomicAdd index & add node index to helper buffer a
//  - set helper buffer b 0
// 4. process parent nodes:
//  - check child nodes and update node
//  - if changed, set parent node in helper buffer b to true
//  - set helper buffer a to 255
// repeat 5. and 6. until root node

//  - if helper buffer b non-zero: update node & collect parent node in helper buffer a
//  - set helper buffer b to 0
// 4. process parent nodes:
//  - check child nodes and update node
//  - if changed, set parent node in helper buffer b to true
//  - set helper buffer a to 255
// 5. process parent node buffer:
//  - if helper buffer b non-zero: atomicAdd index & add node index to helper buffer a
//  - set helper buffer b 0
// repeat 4. and 5. until root node

impl OctreeUpdate {
    pub fn on_brick_cache_updated(
        &self,
        command_encoder: &mut CommandEncoder,
        cache_update_meta: &CacheUpdateMeta,
    ) {
        //log::info!("{:?}", cache_update_meta);

        let indirect_initial_data = vec![DispatchWorkgroupsIndirect::new_1d(); 1];
        for indirect_buffer in self.indirect_buffers.iter() {
            indirect_buffer.write_buffer(indirect_initial_data.as_slice());
        }

        let num_nodes_to_update_initial_data = vec![0];
        for buffer in self.num_nodes_to_update_buffers.iter() {
            buffer.write_buffer(num_nodes_to_update_initial_data.as_slice());
        }

        // todo: all this can be done on the GPU directly if cache management is moved there
        let cache_update_meta_data = vec![CacheUpdateMetaGPU {
            num_mapped: cache_update_meta.mapped_local_brick_ids().len() as u32,
            num_unmapped: cache_update_meta.unmapped_local_brick_ids().len() as u32,
            num_mapped_first_time: cache_update_meta.mapped_first_time_local_brick_ids().len()
                as u32,
            num_unsuccessful_map_attempt: cache_update_meta
                .unsuccessful_map_attempt_local_brick_ids()
                .len() as u32,
        }];
        self.cache_update_meta_buffer
            .write_buffer(cache_update_meta_data.as_slice());
        self.new_brick_ids_buffer.write_buffer(
            cache_update_meta
                .mapped_first_time_local_brick_ids()
                .as_slice(),
        );
        self.mapped_brick_ids_buffer
            .write_buffer(cache_update_meta.mapped_local_brick_ids().as_slice());
        self.unmapped_brick_ids_buffer
            .write_buffer(cache_update_meta.unmapped_local_brick_ids().as_slice());

        // todo: split up into multiple passes for finer timestamp query granularity
 
        {
            // todo: encode on unmapped passes
            let mut update_octree_pass =
                command_encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Label::from("update octree"),
                    timestamp_writes: None,
                });
            self.on_mapped_first_time_passes.encode(
                &mut update_octree_pass,
                cache_update_meta.mapped_first_time_local_brick_ids().len() as u32,
            );
            self.on_mapped_passes.encode(
                &mut update_octree_pass,
                cache_update_meta.mapped_local_brick_ids().len() as u32,
            );
        }
        #[cfg(feature = "timestamp-query")]
        timestamp_query_helper.write_timestamp(command_encoder);
    }
    pub fn copy_to_readable(&self, command_encoder: &mut CommandEncoder) {
        if let Some(break_point) = self.break_point.as_ref() {
            break_point.copy_to_read_buffer(
                command_encoder,
                &self.helper_buffer_a,
                &self.helper_buffer_b,
            );
        }
    }
    pub fn map_break_point(&self) {
        if let Some(break_point) = self.break_point.as_ref() {
            break_point.map();
        }
    }
    pub fn maybe_print_break_point(&self) {
        if let Some(break_point) = self.break_point.as_ref() {
            if break_point.maybe_print() {
                //self.helper_buffer_a.buffer().destroy();
                //self.helper_buffer_b.buffer().destroy();
            }
        }
    }

    pub fn new(
        octree: &Octree,
        volume_manager: &VolumeManager,
        wgsl_preprocessor: &WGSLPreprocessor,
    ) -> Self {
        // todo: clean up
        // todo: set up on unmapped passes

        let gpu = octree.gpu();
        let max_bricks_per_update = volume_manager.brick_transfer_limit();

        let num_nodes_per_subdivision = octree.nodes_per_subdivision();
        let num_leaf_nodes = *num_nodes_per_subdivision.last().unwrap();

        let helper_buffer_a_initial_data = vec![255; num_leaf_nodes];
        let helper_buffer_a = Buffer::from_data(
            "helper buffer a",
            helper_buffer_a_initial_data.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC, // todo: remove copy_src
            octree.gpu(),
        );
        let helper_buffer_b = Buffer::new_zeroed(
            "helper buffer b",
            num_leaf_nodes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC, // todo: remove copy_src
            octree.gpu(),
        );

        let cache_update_meta_buffer = Buffer::new_single_element(
            "cache update meta",
            CacheUpdateMetaGPU::default(),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            octree.gpu(),
        );

        let new_brick_ids_buffer = Buffer::new_zeroed(
            "new brick ids",
            max_bricks_per_update,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            octree.gpu(),
        );
        let mapped_brick_ids_buffer = Buffer::new_zeroed(
            "new brick ids",
            max_bricks_per_update,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            octree.gpu(),
        );
        let unmapped_brick_ids_buffer = Buffer::new_zeroed(
            "new brick ids",
            max_bricks_per_update,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            octree.gpu(),
        );

        // first timers: min max

        let compute_min_max_shader_module =
            gpu.device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                        &*wgsl_preprocessor
                            .preprocess(include_str!("compute_min_max_values.wgsl"))
                            .ok()
                            .unwrap(),
                    )),
                });
        let compute_min_max_pipeline: ComputePipelineData<3> = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("compute min max"),
                layout: None,
                module: &compute_min_max_shader_module,
                entry_point: "main",
            },
            gpu.device(),
        );

        let update_node_min_max_values_shader_module =
            gpu.device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                        &*wgsl_preprocessor
                            .preprocess(include_str!("update_node_min_max_values.wgsl"))
                            .ok()
                            .unwrap(),
                    )),
                });
        let update_node_min_max_values_pipeline: ComputePipelineData<3> = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("update node min max values"),
                layout: None,
                module: &update_node_min_max_values_shader_module,
                entry_point: "main",
            },
            gpu.device(),
        );

        // all mapped bricks

        let process_mapped_bricks_shader_module =
            gpu.device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                        &*wgsl_preprocessor
                            .preprocess(include_str!("process_mapped_bricks.wgsl"))
                            .ok()
                            .unwrap(),
                    )),
                });
        let process_mapped_bricks_pipeline: ComputePipelineData<3> = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("process mapped bricks"),
                layout: None,
                module: &process_mapped_bricks_shader_module,
                entry_point: "main",
            },
            gpu.device(),
        );

        let update_leaf_nodes_shader_module =
            gpu.device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                        &*wgsl_preprocessor
                            .preprocess(include_str!("update_leaf_nodes.wgsl"))
                            .ok()
                            .unwrap(),
                    )),
                });
        let update_leaf_nodes_pipeline: ComputePipelineData<3> = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("update leaf nodes"),
                layout: None,
                module: &update_leaf_nodes_shader_module,
                entry_point: "main",
            },
            gpu.device(),
        );

        // upper levels

        let set_up_next_level_update_shader_module =
            gpu.device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                        &*wgsl_preprocessor
                            .preprocess(include_str!("set_up_next_level_update.wgsl"))
                            .ok()
                            .unwrap(),
                    )),
                });
        let set_up_next_level_update_pipeline: ComputePipelineData<3> = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("set up next level update"),
                layout: None,
                module: &set_up_next_level_update_shader_module,
                entry_point: "main",
            },
            gpu.device(),
        );

        let update_non_leaf_nodes_shader_module =
            gpu.device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                        &*wgsl_preprocessor
                            .preprocess(include_str!("update_non_leaf_nodes.wgsl"))
                            .ok()
                            .unwrap(),
                    )),
                });
        let update_non_leaf_nodes_pipeline: ComputePipelineData<3> = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("update non leaf nodes"),
                layout: None,
                module: &update_non_leaf_nodes_shader_module,
                entry_point: "main",
            },
            gpu.device(),
        );

        let compute_min_max_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("compute min max: bind group 0"),
                layout: compute_min_max_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager
                            .page_table_directory()
                            .page_directory_meta_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: volume_manager
                            .page_table_directory()
                            .page_table_meta_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: volume_manager
                            .page_table_directory()
                            .page_directory_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: volume_manager.lru_cache().cache_as_binding_resource(),
                    },
                ],
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("compute min max: bind group 1"),
                layout: compute_min_max_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: cache_update_meta_buffer.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: new_brick_ids_buffer.buffer().as_entire_binding(),
                    },
                ],
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("compute min max: bind group 2"),
                layout: compute_min_max_pipeline.bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: helper_buffer_a.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: helper_buffer_b.buffer().as_entire_binding(),
                    },
                ],
            });
            DynamicComputeEncodeDescriptor::new(
                compute_min_max_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
            )
        };
        let update_min_max_values_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("update min max: bind group 0"),
                layout: update_node_min_max_values_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager
                            .page_table_directory()
                            .page_directory_meta_as_binding_resource(),
                    },
                ],
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("update min max: bind group 1"),
                layout: update_node_min_max_values_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: helper_buffer_a.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: helper_buffer_b.buffer().as_entire_binding(),
                    },
                ],
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("update min max: bind group 2"),
                layout: update_node_min_max_values_pipeline.bind_group_layout(2),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: octree.octree_nodes_as_binding_resource(),
                }],
            });
            /*
            log::info!(
                "workgroup size = {}, num leaf nodes 0 {}",
                f32::ceil(num_leaf_nodes as f32 / WORKGROUP_SIZE as f32) as u32,
                num_leaf_nodes
            );
             */
            StaticComputeEncodeDescriptor::new_1d(
                update_node_min_max_values_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
                f32::ceil(num_leaf_nodes as f32 / WORKGROUP_SIZE as f32) as u32,
            )
        };
        let min_max_processing_size_3d =
            volume_manager.lru_cache().cache_entry_size() / MIN_MAX_THREAD_BLOCK_SIZE;
        let on_mapped_first_time_passes = OnFirstTimeMappedPasses {
            processing_size: min_max_processing_size_3d.x
                * min_max_processing_size_3d.y
                * min_max_processing_size_3d.z,
            compute_min_max_pass,
            update_min_max_values_pass,
        };

        let indirect_initial_data = vec![DispatchWorkgroupsIndirect::new_1d(); 1];

        let mut subdivision_index_buffers = Vec::new();

        let mut indirect_buffers = Vec::new();
        let mut num_nodes_to_update_buffers = Vec::new();

        let mut on_cache_update_compute_passes = Vec::new();

        let process_mapped_bricks_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("process mapped: bind group 0"),
                layout: process_mapped_bricks_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager
                            .page_table_directory()
                            .page_directory_meta_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: volume_manager
                            .page_table_directory()
                            .page_table_meta_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: octree.octree_nodes_as_binding_resource(),
                    },
                ],
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("process mapped: bind group 1"),
                layout: process_mapped_bricks_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: cache_update_meta_buffer.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: mapped_brick_ids_buffer.buffer().as_entire_binding(),
                    },
                ],
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("process mapped: bind group 2"),
                layout: process_mapped_bricks_pipeline.bind_group_layout(2),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: helper_buffer_a.buffer().as_entire_binding(),
                }],
            });
            DynamicComputeEncodeDescriptor::new(
                process_mapped_bricks_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
            )
        };

        let update_leaf_nodes_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("update leaf nodes: bind group 0"),
                layout: update_leaf_nodes_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager
                            .page_table_directory()
                            .page_directory_meta_as_binding_resource(),
                    },
                ],
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("update leaf nodes: bind group 1"),
                layout: update_leaf_nodes_pipeline.bind_group_layout(1),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: helper_buffer_a.buffer().as_entire_binding(),
                }],
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("update leaf nodes: bind group 2"),
                layout: update_leaf_nodes_pipeline.bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.octree_nodes_as_binding_resource(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: helper_buffer_b.buffer().as_entire_binding(),
                    },
                ],
            });
            ComputePassData::Direct(StaticComputeEncodeDescriptor::new_1d(
                update_leaf_nodes_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
                f32::ceil(num_leaf_nodes as f32 / WORKGROUP_SIZE as f32) as u32,
            ))
        };
        on_cache_update_compute_passes.push(update_leaf_nodes_pass);

        for i in (0..num_nodes_per_subdivision.len() - 1).rev() {
            let num_nodes = num_nodes_per_subdivision[i];
            let stream_compaction_pass_workgroup_size =
                f32::ceil(num_nodes as f32 / WORKGROUP_SIZE as f32) as u32;

            let subdivision_index_buffer = Buffer::new_single_element(
                "subdivision_index",
                i as u32,
                BufferUsages::UNIFORM,
                gpu,
            );

            let indirect_buffer = Rc::new(Buffer::from_data(
                "indirect buffer",
                indirect_initial_data.as_slice(),
                BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
                gpu,
            ));

            let num_nodes_to_update_buffer = Buffer::new_single_element(
                "num nodes next level",
                0u32,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                gpu,
            );

            let stream_compaction_pass = {
                let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("set up next level: bind group 0"),
                    layout: set_up_next_level_update_pipeline.bind_group_layout(0),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: octree.volume_subdivisions_as_binding_resource(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: volume_manager
                                .page_table_directory()
                                .page_directory_meta_as_binding_resource(),
                        },
                    ],
                });
                let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("set up next level: bind group 1"),
                    layout: set_up_next_level_update_pipeline.bind_group_layout(1),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: subdivision_index_buffer.buffer().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: helper_buffer_b.buffer().as_entire_binding(),
                        },
                    ],
                });
                let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("set up next level: bind group 2"),
                    layout: set_up_next_level_update_pipeline.bind_group_layout(2),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: indirect_buffer.buffer().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: num_nodes_to_update_buffer.buffer().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: helper_buffer_a.buffer().as_entire_binding(),
                        },
                    ],
                });
                ComputePassData::Direct(StaticComputeEncodeDescriptor::new_1d(
                    set_up_next_level_update_pipeline.pipeline(),
                    vec![bind_group_0, bind_group_1, bind_group_2],
                    stream_compaction_pass_workgroup_size,
                ))
            };
            let update_non_leaf_nodes_pass = {
                let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("update non leaf node: bind group 0"),
                    layout: update_non_leaf_nodes_pipeline.bind_group_layout(0),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: octree.volume_subdivisions_as_binding_resource(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: volume_manager
                                .page_table_directory()
                                .page_directory_meta_as_binding_resource(),
                        },
                    ],
                });
                let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("update non leaf node: bind group 1"),
                    layout: update_non_leaf_nodes_pipeline.bind_group_layout(1),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: subdivision_index_buffer.buffer().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: num_nodes_to_update_buffer.buffer().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: helper_buffer_a.buffer().as_entire_binding(),
                        },
                    ],
                });
                let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("update non leaf node: bind group 2"),
                    layout: update_non_leaf_nodes_pipeline.bind_group_layout(2),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: octree.octree_nodes_as_binding_resource(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: helper_buffer_b.buffer().as_entire_binding(),
                        },
                    ],
                });
                ComputePassData::Indirect(ComputeEncodeIndirectDescriptor::with_indirect_buffer(
                    update_non_leaf_nodes_pipeline.pipeline(),
                    vec![bind_group_0, bind_group_1, bind_group_2],
                    &indirect_buffer,
                    0,
                ))
            };

            subdivision_index_buffers.push(subdivision_index_buffer);
            indirect_buffers.push(indirect_buffer);
            num_nodes_to_update_buffers.push(num_nodes_to_update_buffer);
            on_cache_update_compute_passes.push(stream_compaction_pass);
            on_cache_update_compute_passes.push(update_non_leaf_nodes_pass);
        }

        let on_mapped_passes = OnMappedPasses {
            process_mapped_bricks_pass,
            dependent_passes: on_cache_update_compute_passes,
        };

        let break_point = None; //Some(BreakPointBuffers::new(&helper_buffer_a, &helper_buffer_b));
        Self {
            gpu: gpu.clone(),
            cache_update_meta_buffer,
            new_brick_ids_buffer,
            mapped_brick_ids_buffer,
            unmapped_brick_ids_buffer,
            subdivision_index_buffers,
            indirect_buffers,
            num_nodes_to_update_buffers,
            helper_buffer_a,
            helper_buffer_b,
            on_mapped_first_time_passes,
            on_mapped_passes,
            break_point,
        }
    }
}
