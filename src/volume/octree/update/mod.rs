use std::rc::Rc;
use wgpu::BufferUsages;
use wgpu_framework::gpu::buffer::Buffer;
use wgsl_preprocessor::WGSLPreprocessor;
use crate::volume::octree::octree_manager::OctreeManager;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CacheUpdateMeta {
    num_mapped: u32,
    num_unmapped: u32,
    num_mapped_first_time: u32,
    num_unsuccessful_map_attempt: u32,
}

#[derive(Clone, Debug)]
struct OctreeUpdate {

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
    fn new(octree: &OctreeManager,
           wgsl_preprocessor: &WGSLPreprocessor,
    ) -> Self {
        let max_bricks_per_update: usize = 32;

        let num_nodes_per_subdivision = octree.nodes_per_subdivision();
        let num_leaf_nodes = *num_nodes_per_subdivision.last().unwrap();

        let helper_buffer_a = Buffer::new_zeroed(
            "helper buffer a",
            num_leaf_nodes,
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
            CacheUpdateMeta::default(),
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

        /*
        let compute_min_max_shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("../ray_cast.wgsl"))
                        .ok()
                        .unwrap(),
                )),
            });

         */

        Self {}
    }
}


