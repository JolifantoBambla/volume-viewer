use crate::resource::sparse_residency::texture3d::brick_cache_update::{
    ChannelBrickCacheUpdateResult, MappedBrick, UnmappedBrick,
};
use crate::util::vec_hash_map::VecHashMap;
use crate::volume::octree::resolution_mapping::ResolutionMapping;
use crate::volume::octree::storage::OctreeStorage;
use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};
use glam::UVec3;
use std::fmt::Debug;
use std::rc::Rc;

pub trait OctreeNode {
    fn from_resolution_mapping(resolution_mapping: usize) -> Self;
}

pub trait PageTableOctree {
    type Node: bytemuck::Pod + Default + OctreeNode + Debug;

    fn create_nodes(&self) -> Vec<Self::Node> {
        let mut nodes = Vec::with_capacity(total_number_of_nodes(self.subdivisions().as_slice()));
        for (octree_level, _) in self.subdivisions().iter().enumerate() {
            nodes.append(&mut vec![Self::Node::from_resolution_mapping(
                self.resolution_mapping().map_to_dataset_level(octree_level),
            )]);
        }
        nodes
    }

    fn new(
        subdivisions: &Rc<Vec<VolumeSubdivision>>,
        data_subdivisions: &Rc<Vec<UVec3>>,
        resolution_mapping: ResolutionMapping,
    ) -> Self;

    fn subdivisions(&self) -> &Rc<Vec<VolumeSubdivision>>;

    fn resolution_mapping(&self) -> &ResolutionMapping;

    fn set_resolution_mapping(&mut self, resolution_mapping: ResolutionMapping);

    fn map_to_highest_subdivision_level(&self, level: usize) -> usize {
        *self
            .resolution_mapping()
            .map_to_octree_subdivision_level(level)
            .last()
            .unwrap_or(&0)
    }

    fn on_brick_cache_update(
        &mut self,
        update_result: &ChannelBrickCacheUpdateResult,
        node_storage: &mut OctreeStorage<Self::Node>,
    ) {
        self.on_mapped_bricks(update_result.mapped_bricks(), node_storage);
        self.on_unmapped_bricks(update_result.unmapped_bricks(), node_storage);
    }

    fn on_mapped_bricks(
        &mut self,
        bricks: &VecHashMap<u32, MappedBrick>,
        node_storage: &mut OctreeStorage<Self::Node>,
    );

    fn on_unmapped_bricks(
        &mut self,
        bricks: &VecHashMap<u32, UnmappedBrick>,
        node_storage: &mut OctreeStorage<Self::Node>,
    );
}
