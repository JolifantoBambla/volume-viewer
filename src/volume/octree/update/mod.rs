use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;
use glam::UVec3;
use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoder, ComputePass, ComputePassDescriptor, ComputePipelineDescriptor, Label};
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::Buffer;
use wgsl_preprocessor::WGSLPreprocessor;
use crate::renderer::pass::{StaticComputeEncodeDescriptor, ComputeEncodeIndirectDescriptor, ComputePassData, ComputePipelineData, DispatchWorkgroupsIndirect, DynamicComputeEncodeDescriptor};
use crate::resource::VolumeManager;
use crate::volume::octree::octree_manager::OctreeManager;

const WORKGROUP_SIZE: u32 = 64;
const MIN_MAX_THREAD_BLOCK_SIZE: UVec3 = UVec3::new(2, 2, 2);

#[derive(Clone, Debug, Default)]
pub struct CacheUpdateMeta {
    mapped_local_brick_ids: Vec<u32>,
    unmapped_local_brick_ids: Vec<u32>,
    mapped_first_time_local_brick_ids: Vec<u32>,
    unsuccessful_map_attempt_local_brick_ids: Vec<u32>,
}

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
            self.compute_min_max_pass.encode_1d(compute_pass, num_new_bricks * self.processing_size / WORKGROUP_SIZE);
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
            self.process_mapped_bricks_pass.encode_1d(compute_pass, num_mapped_bricks / WORKGROUP_SIZE);
            for pass in self.dependent_passes.iter() {
                pass.encode(compute_pass);
            }
        }
    }
}



#[derive(Debug)]
struct OctreeUpdate {
    gpu: Arc<Gpu>,

    // buffers that need updating every frame
    cache_update_meta_buffer: Buffer<CacheUpdateMetaGPU>,
    new_brick_ids_buffer: Buffer<u32>,
    mapped_brick_ids_buffer: Buffer<u32>,
    unmapped_brick_ids_buffer: Buffer<u32>,
    indirect_buffers: Vec<Rc<Buffer<DispatchWorkgroupsIndirect>>>,
    num_nodes_to_update_buffers: Vec<Buffer<u32>>,

    // buffers that must not go out of scope
    helper_buffer_a: Buffer<u32>,
    helper_buffer_b: Buffer<u32>,
    subdivision_index_buffers: Vec<Buffer<u32>>,

    // passes
    on_mapped_first_time_passes: OnFirstTimeMappedPasses,
    on_mapped_passes: OnMappedPasses,
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
    pub fn on_brick_cache_updated(&self, command_encoder: &mut CommandEncoder, cache_update_meta: &CacheUpdateMeta) {
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
            num_mapped: cache_update_meta.mapped_local_brick_ids.len() as u32,
            num_unmapped: cache_update_meta.unmapped_local_brick_ids.len() as u32,
            num_mapped_first_time: cache_update_meta.mapped_first_time_local_brick_ids.len() as u32,
            num_unsuccessful_map_attempt: cache_update_meta.unsuccessful_map_attempt_local_brick_ids.len() as u32
        }];
        self.cache_update_meta_buffer.write_buffer(cache_update_meta_data.as_slice());
        self.new_brick_ids_buffer.write_buffer(cache_update_meta.mapped_first_time_local_brick_ids.as_slice());
        self.mapped_brick_ids_buffer.write_buffer(cache_update_meta.mapped_local_brick_ids.as_slice());
        self.unmapped_brick_ids_buffer.write_buffer(cache_update_meta.unmapped_local_brick_ids.as_slice());

        // todo: encode on unmapped passes

        {
            let mut on_new_bricks_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Label::from("on new bricks pass")
            });
            self.on_mapped_first_time_passes.encode(&mut on_new_bricks_pass, cache_update_meta.mapped_first_time_local_brick_ids.len() as u32);
        }
        {
            let mut on_mapped_bricks_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Label::from("on new bricks pass")
            });
            self.on_mapped_passes.encode(&mut on_mapped_bricks_pass, cache_update_meta.mapped_local_brick_ids.len() as u32);
        }
    }

    fn new(octree: &OctreeManager,
           volume_manager: &VolumeManager,
           wgsl_preprocessor: &WGSLPreprocessor,
    ) -> Self {
        // todo: clean up
        // todo: set up on unmapped passes

        let gpu = octree.gpu();
        let max_bricks_per_update: usize = 32;

        let num_nodes_per_subdivision = octree.nodes_per_subdivision();
        let num_leaf_nodes = *num_nodes_per_subdivision.last().unwrap();

        let helper_buffer_a_initial_data = vec![255; num_leaf_nodes];
        let helper_buffer_a = Buffer::from_data(
            "helper buffer a",
            helper_buffer_a_initial_data.as_slice(),
            BufferUsages::STORAGE,
            octree.gpu()
        );
        let helper_buffer_b = Buffer::new_zeroed(
            "helper buffer b",
            num_leaf_nodes,
            BufferUsages::STORAGE,
            octree.gpu()
        );

        let cache_update_meta_buffer = Buffer::new_single_element(
            "cache update meta",
            CacheUpdateMetaGPU::default(),
            BufferUsages::STORAGE,
            octree.gpu()
        );

        let new_brick_ids_buffer = Buffer::new_zeroed(
            "new brick ids",
            max_bricks_per_update,
            BufferUsages::STORAGE,
            octree.gpu()
        );
        let mapped_brick_ids_buffer = Buffer::new_zeroed(
            "new brick ids",
            max_bricks_per_update,
            BufferUsages::STORAGE,
            octree.gpu()
        );
        let unmapped_brick_ids_buffer = Buffer::new_zeroed(
            "new brick ids",
            max_bricks_per_update,
            BufferUsages::STORAGE,
            octree.gpu()
        );

        // first timers: min max

        let compute_min_max_shader_module = gpu
            .device()
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
                module : &compute_min_max_shader_module,
                entry_point: "main"
            },
            gpu.device(),
        );

        let update_node_min_max_values_shader_module = gpu
            .device()
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
                module : &update_node_min_max_values_shader_module,
                entry_point: "main"
            },
            gpu.device(),
        );

        // all mapped bricks

        let process_mapped_bricks_shader_module = gpu
            .device()
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
                module : &process_mapped_bricks_shader_module,
                entry_point: "main"
            },
            gpu.device(),
        );

        let update_leaf_nodes_shader_module = gpu
            .device()
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
                module : &update_leaf_nodes_shader_module,
                entry_point: "main"
            },
            gpu.device(),
        );

        // upper levels

        let set_up_next_level_update_shader_module = gpu
            .device()
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
                module : &set_up_next_level_update_shader_module,
                entry_point: "main"
            },
            gpu.device(),
        );

        let update_non_leaf_nodes_shader_module = gpu
            .device()
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
                module : &update_non_leaf_nodes_shader_module,
                entry_point: "main"
            },
            gpu.device(),
        );

        let compute_min_max_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 0"),
                layout: &compute_min_max_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager.page_table_directory().page_directory_meta_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: volume_manager.page_table_directory().page_table_meta_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: volume_manager.page_table_directory().page_directory_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: volume_manager.lru_cache().cache_as_binding_resource()
                    }
                ]
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 1"),
                layout: &compute_min_max_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: cache_update_meta_buffer.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: new_brick_ids_buffer.buffer().as_entire_binding()
                    }
                ]
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 2"),
                layout: &compute_min_max_pipeline.bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: helper_buffer_a.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: helper_buffer_b.buffer().as_entire_binding()
                    }
                ]
            });
            DynamicComputeEncodeDescriptor::new(
                compute_min_max_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
            )
        };
        let update_min_max_values_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 0"),
                layout: &update_node_min_max_values_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager.page_table_directory().page_directory_meta_as_binding_resource()
                    }
                ]
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 1"),
                layout: &update_node_min_max_values_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: helper_buffer_a.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: helper_buffer_b.buffer().as_entire_binding()
                    }
                ]
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 2"),
                layout: &update_node_min_max_values_pipeline.bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.octree_nodes_as_binding_resource()
                    }
                ]
            });
            StaticComputeEncodeDescriptor::new_1d(
                update_node_min_max_values_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
                num_leaf_nodes as u32 / WORKGROUP_SIZE,
            )
        };
        let min_max_processing_size_3d = volume_manager.lru_cache().cache_entry_size() / MIN_MAX_THREAD_BLOCK_SIZE;
        let on_mapped_first_time_passes = OnFirstTimeMappedPasses {
            processing_size: min_max_processing_size_3d.x * min_max_processing_size_3d.y * min_max_processing_size_3d.z,
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
                label: Label::from("bind group 0"),
                layout: &process_mapped_bricks_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager.page_table_directory().page_directory_meta_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: volume_manager.page_table_directory().page_table_meta_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: volume_manager.page_table_directory().page_directory_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: volume_manager.lru_cache().cache_as_binding_resource()
                    }
                ]
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 1"),
                layout: &process_mapped_bricks_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: cache_update_meta_buffer.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: mapped_brick_ids_buffer.buffer().as_entire_binding()
                    }
                ]
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 2"),
                layout: &process_mapped_bricks_pipeline.bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: helper_buffer_b.buffer().as_entire_binding()
                    }
                ]
            });
            DynamicComputeEncodeDescriptor::new(
                process_mapped_bricks_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
            )
        };
        let first_subdivision_index_buffer = Buffer::new_single_element(
            "subdivision_index",
            1 as u32,
            BufferUsages::UNIFORM,
            gpu
        );
        let first_indirect_buffer = Rc::new(Buffer::from_data(
            "indirect buffer",
            indirect_initial_data.as_slice(),
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
            gpu
        ));
        let first_num_nodes_to_update_buffer = Buffer::new_single_element(
            "num nodes next level",
            0u32,
            BufferUsages::STORAGE,
            gpu
        );
        let update_leaf_nodes_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 0"),
                layout: &update_leaf_nodes_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager.page_table_directory().page_directory_meta_as_binding_resource()
                    }
                ]
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 1"),
                layout: &update_leaf_nodes_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: helper_buffer_b.buffer().as_entire_binding()
                    }
                ]
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 2"),
                layout: &update_leaf_nodes_pipeline.bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.octree_nodes_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: first_indirect_buffer.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: first_num_nodes_to_update_buffer.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: helper_buffer_a.buffer().as_entire_binding()
                    }
                ]
            });
            ComputePassData::Direct(StaticComputeEncodeDescriptor::new_1d(
                update_leaf_nodes_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
                num_leaf_nodes as u32 / WORKGROUP_SIZE,
            ))
        };
        on_cache_update_compute_passes.push(update_leaf_nodes_pass);

        let update_first_parent_level_pass = {
            let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 0"),
                layout: &update_non_leaf_nodes_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.volume_subdivisions_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: volume_manager.page_table_directory().page_directory_meta_as_binding_resource()
                    }
                ]
            });
            let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 1"),
                layout: &update_non_leaf_nodes_pipeline.bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: first_subdivision_index_buffer.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: first_num_nodes_to_update_buffer.buffer().as_entire_binding()
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: helper_buffer_a.buffer().as_entire_binding()
                    }
                ]
            });
            let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Label::from("bind group 2"),
                layout: &update_non_leaf_nodes_pipeline.bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: octree.octree_nodes_as_binding_resource()
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: helper_buffer_b.buffer().as_entire_binding()
                    }
                ]
            });
            ComputePassData::Indirect(ComputeEncodeIndirectDescriptor::with_indirect_buffer(
                update_non_leaf_nodes_pipeline.pipeline(),
                vec![bind_group_0, bind_group_1, bind_group_2],
                &first_indirect_buffer,
                0
            ))
        };
        on_cache_update_compute_passes.push(update_first_parent_level_pass);

        subdivision_index_buffers.push(first_subdivision_index_buffer);
        indirect_buffers.push(first_indirect_buffer);
        num_nodes_to_update_buffers.push(first_num_nodes_to_update_buffer);

        for i in (2..num_nodes_per_subdivision.len()).rev() {
            let num_nodes = num_nodes_per_subdivision[i];
            let stream_compaction_pass_workgroup_size = num_nodes as u32 / WORKGROUP_SIZE;

            let subdivision_index_buffer = Buffer::new_single_element(
                "subdivision_index",
                i as u32,
                BufferUsages::UNIFORM,
                gpu
            );

            let indirect_buffer = Rc::new(Buffer::from_data(
                "indirect buffer",
                indirect_initial_data.as_slice(),
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                gpu
            ));

            let num_nodes_to_update_buffer = Buffer::new_single_element(
                "num nodes next level",
                0u32,
                BufferUsages::STORAGE,
                gpu
            );

            let stream_compaction_pass = {
                let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("bind group 0"),
                    layout: &set_up_next_level_update_pipeline.bind_group_layout(0),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: octree.volume_subdivisions_as_binding_resource()
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: volume_manager.page_table_directory().page_directory_meta_as_binding_resource()
                        }
                    ]
                });
                let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("bind group 1"),
                    layout: &set_up_next_level_update_pipeline.bind_group_layout(1),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: subdivision_index_buffer.buffer().as_entire_binding()
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: helper_buffer_b.buffer().as_entire_binding()
                        }
                    ]
                });
                let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("bind group 2"),
                    layout: &set_up_next_level_update_pipeline.bind_group_layout(2),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: indirect_buffer.buffer().as_entire_binding()
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: num_nodes_to_update_buffer.buffer().as_entire_binding()
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: helper_buffer_a.buffer().as_entire_binding()
                        }
                    ]
                });
                ComputePassData::Direct(StaticComputeEncodeDescriptor::new_1d(
                    set_up_next_level_update_pipeline.pipeline(),
                    vec![bind_group_0, bind_group_1, bind_group_2],
                    stream_compaction_pass_workgroup_size,
                ))
            };
            let update_non_leaf_nodes_pass = {
                let bind_group_0 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("bind group 0"),
                    layout: &update_non_leaf_nodes_pipeline.bind_group_layout(0),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: octree.volume_subdivisions_as_binding_resource()
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: volume_manager.page_table_directory().page_directory_meta_as_binding_resource()
                        }
                    ]
                });
                let bind_group_1 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("bind group 1"),
                    layout: &update_non_leaf_nodes_pipeline.bind_group_layout(1),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: subdivision_index_buffer.buffer().as_entire_binding()
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: num_nodes_to_update_buffer.buffer().as_entire_binding()
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: helper_buffer_a.buffer().as_entire_binding()
                        }
                    ]
                });
                let bind_group_2 = gpu.device().create_bind_group(&BindGroupDescriptor {
                    label: Label::from("bind group 2"),
                    layout: &update_non_leaf_nodes_pipeline.bind_group_layout(2),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: octree.octree_nodes_as_binding_resource()
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: helper_buffer_b.buffer().as_entire_binding()
                        }
                    ]
                });
                ComputePassData::Indirect(ComputeEncodeIndirectDescriptor::with_indirect_buffer(
                    update_non_leaf_nodes_pipeline.pipeline(),
                    vec![bind_group_0, bind_group_1, bind_group_2],
                    &indirect_buffer,
                    0
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
            dependent_passes: on_cache_update_compute_passes
        };

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
        }
    }
}
