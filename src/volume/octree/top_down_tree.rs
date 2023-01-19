use crate::resource::sparse_residency::texture3d::brick_cache_update::{
    MappedBrick, UnmappedBrick,
};
use crate::util::extent::ToNormalizedAddress;
use crate::util::vec_hash_map::VecHashMap;
use crate::volume::octree::page_table_octree::{OctreeNode, PageTableOctree};
use crate::volume::octree::resolution_mapping::ResolutionMapping;
use crate::volume::octree::storage::OctreeStorage;
use crate::volume::octree::subdivision::VolumeSubdivision;
use crate::volume::BrickAddress;
use glam::{UVec3, Vec3};
use modular_bitfield::prelude::*;
use std::rc::Rc;
use crate::volume::octree::subdivision;

#[bitfield]
#[repr(u8)]
#[derive(Copy, Clone, Debug, Default)]
struct MappedState {
    mapped: bool,
    resolution_mapping: B7,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    /// A bitfield that stores for each of this node's subtrees this bitfield stores if it is
    /// partially mapped or not.
    /// Note that a subtree is also considered mapped if it is known to be empty or homogenous.
    partially_mapped_subtrees: u8,

    /// Stores if this node is mapped or not in its first bit and which resolution in the data set
    /// this node maps to in the remaining 7 bits.
    self_mapped_and_resolution_mapping: u8,

    /// The minimum value in the region in the volume represented by this node.
    min: u8,

    /// The maximum value in the region in the volume represented by this node.
    max: u8,
}

impl Node {
    pub fn is_mapped(&self) -> bool {
        MappedState::from(self.self_mapped_and_resolution_mapping).mapped()
    }

    pub fn set_mapped(&mut self, min: u8, max: u8) {
        let mapped_state =
            MappedState::from(self.self_mapped_and_resolution_mapping).with_mapped(true);
        self.self_mapped_and_resolution_mapping = u8::from(mapped_state);
        self.min = min;
        self.max = max;
    }

    /// Possibly marks a node as unmapped and returns true if it did so and none of the node's
    /// subtrees are partially mapped.
    /// A node is not unmapped if its `min` and `max` are the same, i.e., the node is empty or
    /// homogeneous.
    pub fn set_unmapped(&mut self) -> bool {
        // todo: maybe some average threshold?
        let unmapped = self.min != self.max;
        if unmapped {
            let mapped_state =
                MappedState::from(self.self_mapped_and_resolution_mapping).with_mapped(false);
            self.self_mapped_and_resolution_mapping = u8::from(mapped_state);
        }
        unmapped && !self.has_partially_mapped_subtrees()
    }

    pub fn maps_to_resolution(&self) -> usize {
        MappedState::from(self.self_mapped_and_resolution_mapping).resolution_mapping() as usize
    }

    /// Marks the subtree referenced by `subtree_index` as partially mapped.
    /// Returns `true` if it is the first of this node's partially mapped subtrees.
    pub fn set_subtree_mapped(&mut self, subtree_index: u8) -> bool {
        let had_no_mapped_subtrees = !self.has_partially_mapped_subtrees();
        self.partially_mapped_subtrees |= subtree_index;
        had_no_mapped_subtrees
    }

    /// Marks the subtree referenced by `subtree_index` as unmapped.
    /// Returns true if none of the node's subtrees are mapped at the end of this operation.
    pub fn set_subtree_unmapped(&mut self, subtree_bitmask: u8) -> bool {
        if self.is_subtree_partially_mapped(subtree_bitmask) {
            self.partially_mapped_subtrees -= subtree_bitmask;
        }
        !self.has_partially_mapped_subtrees()
    }

    pub fn is_subtree_partially_mapped(&self, subtree_bitmask: u8) -> bool {
        self.partially_mapped_subtrees & subtree_bitmask > 0
    }

    pub fn has_partially_mapped_subtrees(&self) -> bool {
        self.partially_mapped_subtrees > 0
    }

    pub fn all_subtrees_partially_mapped(&self, subdivision: &VolumeSubdivision) -> bool {
        self.partially_mapped_subtrees == subdivision.full_subtree_mask() as u8
    }

    pub fn set_from(&mut self, other: Self) {
        self.partially_mapped_subtrees = other.partially_mapped_subtrees;
        self.self_mapped_and_resolution_mapping = other.self_mapped_and_resolution_mapping;
        self.min = other.min;
        self.max = other.max;
    }
}

impl OctreeNode for Node {
    fn from_resolution_mapping(resolution_mapping: usize) -> Self {
        let mapped_state = MappedState::new()
            .with_mapped(false)
            .with_resolution_mapping(resolution_mapping as u8);
        Self {
            self_mapped_and_resolution_mapping: u8::from(mapped_state),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct TopDownTree {
    subdivisions: Rc<Vec<VolumeSubdivision>>,
    data_subdivisions: Rc<Vec<UVec3>>,
    resolution_mapping: ResolutionMapping,
}

impl TopDownTree {
    // todo: move this into PageTableOctree and only access nodes through this (allows for implementation backed by one large array of nodes for all channels & also for implementation backed by one array of nodes for each channel)
    /// Maps a brick address to a normalized address (i.e., in the unit cube).
    fn to_normalized_address(&self, brick_address: BrickAddress) -> Vec3 {
        brick_address.index.to_normalized_address(
            self.data_subdivisions
                .get(brick_address.level as usize)
                .unwrap(),
        )
    }
}

impl PageTableOctree for TopDownTree {
    type Node = Node;

    fn new(
        subdivisions: &Rc<Vec<VolumeSubdivision>>,
        data_subdivisions: &Rc<Vec<UVec3>>,
        resolution_mapping: ResolutionMapping,
    ) -> Self {
        Self {
            subdivisions: subdivisions.clone(),
            data_subdivisions: data_subdivisions.clone(),
            resolution_mapping,
        }
    }

    fn subdivisions(&self) -> &Rc<Vec<VolumeSubdivision>> {
        &self.subdivisions
    }

    fn resolution_mapping(&self) -> &ResolutionMapping {
        &self.resolution_mapping
    }

    fn set_resolution_mapping(&mut self, resolution_mapping: ResolutionMapping, node_storage: &mut OctreeStorage<Self::Node>) {
        // todo: update all nodes where the resolution mapping changed

        // between the same two levels in the data set, the cut between octree levels mapping to
        // one or the other will always be between the same two octree levels
        // since resolution ranges are continuous, i.e., if two data set resolution levels a & b
        // where a < b are used, every level c, a < c < b is also used,
        // there are only two cases that we need to handle:
        //  - a lower resolution is no longer available (map to next higher resolution)
        //  - a higher resolution is no longer available (map to next lower resolution)
        // this means, we only need to look at the boundaries and check if they changed,
        // if they changed

        let old_min = self.resolution_mapping.min_dataset_level();
        let old_max = self.resolution_mapping.max_dataset_level();

        // we iterate from the highest resolution to the lowest
        for dataset_level in (resolution_mapping.min_dataset_level()..resolution_mapping.max_dataset_level() + 1).rev() {
            let new_mapping = resolution_mapping.map_to_octree_subdivision_level(dataset_level).unwrap();
            let next_higher_level = new_mapping.last().unwrap() + 1;

            let new_lower_resolution = dataset_level < old_min;
            let new_higher_resolution = dataset_level > old_max;
            let is_highest_resolution = self.subdivisions.len() <= next_higher_level;
            let needs_unmapping = new_higher_resolution || (new_lower_resolution && is_highest_resolution);

            if needs_unmapping {
                for octree_level in new_mapping.iter().rev() {
                    for node_index in self.subdivisions().get(*octree_level).unwrap().node_indices() {
                        node_storage.node_mut(node_index)
                            .unwrap()
                            .set_from(Node::from_resolution_mapping(dataset_level))
                    }
                }
            } else if new_lower_resolution {
                // every node on this level is unmapped now but the subtrees might be mapped
                // since we went from the highest resolution down to this one, we can use the information
                // in the next higher level outside of the range we're iterating over
                // or maybe we can even use the data stored in the nodes themselves? no we can't -> it might be that we're dealing with a complete new set of resolutions

                for octree_level in new_mapping.iter().rev() {
                    // todo: set node data from level above
                }
            } else {
                // todo: check if anything needs to be done
                let old_mapping = self.resolution_mapping.map_to_octree_subdivision_level(dataset_level).unwrap();
                let old_lower_boundary = old_mapping.first().unwrap();
                let new_lower_boundary = new_mapping.first().unwrap();
                let old_higher_boundary = old_mapping.last().unwrap();
                let new_higher_boundary = new_mapping.last().unwrap();

                if old_higher_boundary != new_higher_boundary {
                    // the range will be zero if new_higher_boundary is lower than old_higher_boundary
                    // which is what I want, I think
                    for octree_level in (*old_higher_boundary..*new_higher_boundary + 1).rev() {
                        // todo: set node data from level above
                    }
                }
                if old_lower_boundary != new_lower_boundary {
                    for octree_level in (0..0).rev() {
                        // todo:
                    }
                }
            }
        }

        self.resolution_mapping = resolution_mapping;
    }

    fn on_mapped_bricks(
        &mut self,
        bricks: &VecHashMap<u32, MappedBrick>,
        node_storage: &mut OctreeStorage<Self::Node>,
    ) {
        // iterate over resolution levels from min to max (i.e., highest res to lowest res)
        for (&resolution_level, bricks) in bricks.iter() {
            let octree_level = self.resolution_mapping()
                .map_to_highest_subdivision_level(resolution_level as usize);

            for b in bricks.iter() {
                let normalized_address = self.to_normalized_address(b.global_address);

                // set the node as mapped
                let subdivision = self.subdivisions.get(octree_level).unwrap();
                let node_index = subdivision.to_node_index(normalized_address);
                node_storage
                    .node_mut(node_index)
                    .unwrap()
                    .set_mapped(b.min, b.max);

                let mut all_children_maybe_mapped = true;
                let mut child_node_index = node_index;
                for l in (0..octree_level - 1).rev() {
                    let level_subdivision = self.subdivisions.get(l).unwrap();
                    let level_node_index = level_subdivision.to_node_index(normalized_address);
                    let subtree_mask =
                        level_subdivision.subtree_mask(level_node_index, child_node_index) as u8;

                    // For each virtual subdivision level that maps to the same subdivision as the
                    // brick, we may need to set the corresponding node to a mapped state if all of
                    // its child nodes are mapped.
                    // If any node on the path to the next lower subdivision level has unmapped
                    // children, this process can stop.
                    let dataset_level = self.resolution_mapping.map_to_dataset_level(l) as u32;
                    let is_virtual_child_of_brick_level = dataset_level == resolution_level;
                    if is_virtual_child_of_brick_level
                        && all_children_maybe_mapped
                        && !node_storage.node(level_node_index).unwrap().is_mapped()
                    {
                        let mut all_children_mapped = true;
                        let mut min = b.min;
                        let mut max = b.max;
                        for child_index in level_subdivision.child_node_indices(level_node_index) {
                            let child_node = node_storage.node(child_index).unwrap();
                            all_children_mapped = all_children_mapped && child_node.is_mapped();
                            if !all_children_mapped {
                                break;
                            }
                            min = min.min(child_node.min);
                            max = max.max(child_node.max);
                        }
                        all_children_maybe_mapped = all_children_mapped;
                        if all_children_mapped {
                            node_storage
                                .node_mut(level_node_index)
                                .unwrap()
                                .set_mapped(min, max);
                        }
                        // if one of the node's subtrees was already partially mapped and not all of
                        // the node's children are now mapped, we can terminate the loop
                        if !node_storage
                            .node_mut(level_node_index)
                            .unwrap()
                            .set_subtree_mapped(subtree_mask)
                            && !all_children_mapped
                        {
                            break;
                        }
                    } else {
                        // We set the node's subtree as partially mapped.
                        // We can stop this iteration as soon as a node already had partially mapped
                        // subtree before this operation since all levels above will already have been mapped
                        if !node_storage
                            .node_mut(level_node_index)
                            .unwrap()
                            .set_subtree_mapped(subtree_mask)
                        {
                            break;
                        }
                    }
                    child_node_index = level_node_index;
                }
            }
        }
    }

    fn on_unmapped_bricks(
        &mut self,
        bricks: &VecHashMap<u32, UnmappedBrick>,
        node_storage: &mut OctreeStorage<Self::Node>,
    ) {
        // iterate over resolution levels from min to max (i.e., highest res to lowest res)
        for (&resolution_level, bricks) in bricks.iter() {
            let octree_level = self.resolution_mapping()
                .map_to_highest_subdivision_level(resolution_level as usize);

            for b in bricks.iter() {
                let normalized_address = self.to_normalized_address(b.global_address);

                let subdivision = self.subdivisions.get(octree_level).unwrap();
                let node_index = subdivision.to_node_index(normalized_address);

                // maybe set the node as unmapped (homogeneous nodes are ignored)
                // if the node has been unmapped, propagate this change to the upper levels
                if node_storage.node_mut(node_index).unwrap().set_unmapped() {
                    let mut child_node_index = node_index;
                    for l in (0..octree_level - 1).rev() {
                        let level_subdivision = self.subdivisions.get(l).unwrap();
                        let level_node_index = level_subdivision.to_node_index(normalized_address);
                        let subtree_mask = level_subdivision
                            .subtree_mask(level_node_index, child_node_index)
                            as u8;

                        // set the node's subtree as unmapped.
                        // if the node still has mapped nodes, we can terminate the process
                        if !node_storage
                            .node_mut(level_node_index)
                            .unwrap()
                            .set_subtree_unmapped(subtree_mask)
                        {
                            break;
                        }
                        child_node_index = level_node_index;
                    }
                }
            }
        }
    }
}
