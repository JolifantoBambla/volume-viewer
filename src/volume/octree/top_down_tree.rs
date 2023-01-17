use crate::util::extent::ToNormalizedAddress;
use crate::util::vec_hash_map::VecHashMap;
use crate::volume::octree::subdivision::VolumeSubdivision;
use crate::volume::octree::{
    BrickCacheUpdateListener, MappedBrick, PageTableOctree, PageTableOctreeNode, ResolutionMapping,
    UnmappedBrick,
};
use crate::volume::BrickAddress;
use glam::{UVec3, Vec3};
use modular_bitfield::prelude::*;
use std::ops::Range;
use std::rc::Rc;

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

    /// Marks the subtree referenced by `subtree_index` as partially mapped.
    /// Returns `true` if it is the first of this node's partially mapped subtrees.
    pub fn set_subtree_mapped(&mut self, subtree_index: u8) -> bool {
        let had_no_mapped_subtrees = !self.has_partially_mapped_subtrees();
        self.partially_mapped_subtrees |= subtree_index;
        had_no_mapped_subtrees
    }

    /// Marks the subtree referenced by `subtree_index` as unmapped.
    /// Returns true if none of the node's subtrees are mapped at the end of this operation.
    pub fn set_subtree_unmapped(&mut self, subtree_index: u8) -> bool {
        if self.is_subtree_partially_mapped(subtree_index) {
            self.partially_mapped_subtrees -= subtree_index;
        }
        !self.has_partially_mapped_subtrees()
    }

    pub fn is_subtree_partially_mapped(&self, subtree_index: u8) -> bool {
        self.partially_mapped_subtrees & subtree_index > 0
    }

    pub fn has_partially_mapped_subtrees(&self) -> bool {
        self.partially_mapped_subtrees > 0
    }

    pub fn all_subtrees_partially_mapped(&self) -> bool {
        self.partially_mapped_subtrees == u8::MAX
    }
}

impl PageTableOctreeNode for Node {
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
    nodes: Vec<Node>,
    subdivisions: Rc<Vec<VolumeSubdivision>>,
    data_subdivisions: Rc<Vec<UVec3>>,
    resolution_mapping: ResolutionMapping,
}

impl TopDownTree {
    fn subtree_index(&self, node_index: usize, child_node_index: usize) -> usize {
        1 << (child_node_index - node_index * 8)
    }

    // todo: move this into PageTableOctree and only access nodes through this (allows for implementation backed by one large array of nodes for all channels & also for implementation backed by one array of nodes for each channel)
    fn first_child_node_index(&self, child_level_offset: usize, node_index: usize) -> usize {
        child_level_offset + node_index * 8
    }

    fn child_node_indices(&self, child_level_offset: usize, node_index: usize) -> Range<usize> {
        let first_child_node_index = self.first_child_node_index(child_level_offset, node_index);
        first_child_node_index..first_child_node_index + 8
    }

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
            nodes: Self::create_nodes_from_subdivisions(
                subdivisions.as_slice(),
                &resolution_mapping,
            ),
            subdivisions: subdivisions.clone(),
            data_subdivisions: data_subdivisions.clone(),
            resolution_mapping,
        }
    }

    fn nodes(&self) -> &Vec<Self::Node> {
        &self.nodes
    }

    fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    fn node(&self, node_index: usize) -> Option<&Node> {
        self.nodes.get(node_index)
    }

    fn node_mut(&mut self, node_index: usize) -> Option<&mut Node> {
        self.nodes.get_mut(node_index)
    }

    fn subdivisions(&self) -> &Rc<Vec<VolumeSubdivision>> {
        &self.subdivisions
    }

    fn resolution_mapping(&self) -> &ResolutionMapping {
        &self.resolution_mapping
    }

    fn set_resolution_mapping(&mut self, resolution_mapping: ResolutionMapping) {
        // todo: update all nodes
        self.resolution_mapping = resolution_mapping;
    }
}

impl BrickCacheUpdateListener for TopDownTree {
    fn on_mapped_bricks(&mut self, bricks: &VecHashMap<u32, MappedBrick>) {
        // iterate over resolution levels from min to max (i.e., highest res to lowest res)
        for (&resolution_level, bricks) in bricks.iter() {
            let octree_level = self.map_to_highest_subdivision_level(resolution_level as usize);

            for b in bricks.iter() {
                let normalized_address = self.to_normalized_address(b.global_address);

                // set the node as mapped
                let subdivision = self.subdivisions.get(octree_level).unwrap();
                let node_index = subdivision.to_node_index(normalized_address);
                self.node_mut(node_index).unwrap().set_mapped(b.min, b.max);

                let mut all_children_maybe_mapped = true;
                let mut child_node_index = node_index;
                for l in (0..octree_level - 1).rev() {
                    let level_subdivision = self.subdivisions.get(l).unwrap();
                    let level_node_index = level_subdivision.to_node_index(normalized_address);
                    let subtree_index =
                        self.subtree_index(level_node_index, child_node_index) as u8;

                    // For each virtual subdivision level that maps to the same subdivision as the
                    // brick, we may need to set the corresponding node to a mapped state if all of
                    // its child nodes are mapped.
                    // If any node on the path to the next lower subdivision level has unmapped
                    // children, this process can stop.
                    let dataset_level = self.resolution_mapping.map_to_dataset_level(l) as u32;
                    let is_virtual_child_of_brick_level = dataset_level == resolution_level;
                    if is_virtual_child_of_brick_level
                        && all_children_maybe_mapped
                        && !self.node(level_node_index).unwrap().is_mapped()
                    {
                        let mut all_children_mapped = true;
                        let mut min = b.min;
                        let mut max = b.max;
                        for child_index in self.child_node_indices(
                            level_subdivision.next_subdivision_offset() as usize,
                            level_node_index,
                        ) {
                            let child_node = self.node(child_index).unwrap();
                            all_children_mapped = all_children_mapped && child_node.is_mapped();
                            if !all_children_mapped {
                                break;
                            }
                            min = min.min(child_node.min);
                            max = max.max(child_node.max);
                        }
                        all_children_maybe_mapped = all_children_mapped;
                        if all_children_mapped {
                            self.node_mut(level_node_index)
                                .unwrap()
                                .set_mapped(min, max);
                        }
                        // if one of the node's subtrees was already partially mapped and not all of
                        // the node's children are now mapped, we can terminate the loop
                        if !self
                            .node_mut(level_node_index)
                            .unwrap()
                            .set_subtree_mapped(subtree_index)
                            && !all_children_mapped
                        {
                            break;
                        }
                    } else {
                        // We set the node's subtree as partially mapped.
                        // We can stop this iteration as soon as a node already had partially mapped
                        // subtree before this operation since all levels above will already have been mapped
                        if !self
                            .node_mut(level_node_index)
                            .unwrap()
                            .set_subtree_mapped(subtree_index)
                        {
                            break;
                        }
                    }
                    child_node_index = level_node_index;
                }
            }
        }
    }

    fn on_unmapped_bricks(&mut self, bricks: &VecHashMap<u32, UnmappedBrick>) {
        // iterate over resolution levels from min to max (i.e., highest res to lowest res)
        for (&resolution_level, bricks) in bricks.iter() {
            let octree_level = self.map_to_highest_subdivision_level(resolution_level as usize);

            for b in bricks.iter() {
                let normalized_address = self.to_normalized_address(b.global_address);

                let subdivision = self.subdivisions.get(octree_level).unwrap();
                let node_index = subdivision.to_node_index(normalized_address);

                // maybe set the node as unmapped (homogeneous nodes are ignored)
                // if the node has been unmapped, propagate this change to the upper levels
                if self.node_mut(node_index).unwrap().set_unmapped() {
                    let mut child_node_index = node_index;
                    for l in (0..octree_level - 1).rev() {
                        let level_subdivision = self.subdivisions.get(l).unwrap();
                        let level_node_index = level_subdivision.to_node_index(normalized_address);
                        let subtree_index =
                            self.subtree_index(level_node_index, child_node_index) as u8;

                        // set the node's subtree as unmapped.
                        // if the node still has mapped nodes, we can terminate the process
                        if !self
                            .node_mut(level_node_index)
                            .unwrap()
                            .set_subtree_unmapped(subtree_index)
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
