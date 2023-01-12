use crate::util::extent::{ToNormalizedAddress, ToSubscript};
use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};
use crate::volume::octree::{
    BrickCacheUpdateListener, MappedBrick, PageTableOctree, ResolutionMapping, UnmappedBrick,
};
use glam::UVec3;
use std::rc::Rc;
use crate::util::vec_hash_map::VecHashMap;

/*
#[modular_bitfield::bitfield]
#[repr(u8)]
#[derive(Copy, Clone, Debug, Default)]
struct MappedState {
    subtree_0_partially_mapped: bool,
    subtree_1_partially_mapped: bool,
    subtree_2_partially_mapped: bool,
    subtree_3_partially_mapped: bool,
    subtree_4_partially_mapped: bool,
    subtree_5_partially_mapped: bool,
    subtree_6_partially_mapped: bool,
    subtree_7_partially_mapped: bool,
}
*/

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    /// A bitfield that stores for each of this node's subtrees this bitfield stores if it is
    /// partially mapped or not.
    /// Note that a subtree is also considered mapped if it is known to be empty or homogenous.
    partially_mapped_subtrees: u8,

    /// Stores if this node is mapped or not and some other data or padding.
    /// Possibly: The average value in the region in the volume represented by this node.
    self_mapped_and_padding: u8,

    /// The minimum value in the region in the volume represented by this node.
    min: u8,

    /// The maximum value in the region in the volume represented by this node.
    max: u8,
}

impl Node {
    pub fn set_mapped(&mut self, min: u8, max: u8) {
        self.self_mapped_and_padding = true as u8;
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
            self.self_mapped_and_padding = false as u8;
        }
        unmapped && !self.has_mapped_subtrees()
    }

    /// Marks the subtree referenced by `subtree_index` as partially mapped.
    /// Returns `true` if it is the first of this node's partially mapped subtrees.
    pub fn set_subtree_mapped(&mut self, subtree_index: u8) -> bool {
        let had_no_mapped_subtrees = !self.has_mapped_subtrees();
        self.partially_mapped_subtrees |= subtree_index;
        had_no_mapped_subtrees
    }

    /// Marks the subtree referenced by `subtree_index` as unmapped.
    /// Returns true if none of the node's subtrees are mapped at the end of this operation.
    pub fn set_subtree_unmapped(&mut self, subtree_index: u8) -> bool {
        if self.is_subtree_mapped(subtree_index) {
            self.partially_mapped_subtrees -= subtree_index;
        }
        !self.has_mapped_subtrees()
    }

    pub fn is_subtree_mapped(&self, subtree_index: u8) -> bool {
        self.partially_mapped_subtrees & subtree_index > 0
    }

    pub fn has_mapped_subtrees(&self) -> bool {
        self.partially_mapped_subtrees > 0
    }
}

#[derive(Clone, Debug)]
pub struct TopDownTree {
    nodes: Vec<Node>,
    subdivisions: Rc<Vec<VolumeSubdivision>>,
    data_subdivisions: Rc<Vec<UVec3>>,
    resolution_mapping: ResolutionMapping,
}

impl PageTableOctree for TopDownTree {
    type Node = Node;

    fn new(
        subdivisions: &Rc<Vec<VolumeSubdivision>>,
        data_subdivisions: &Rc<Vec<UVec3>>,
        resolution_mapping: ResolutionMapping,
    ) -> Self {
        Self {
            nodes: Self::create_nodes_from_subdivisions(subdivisions.as_slice()),
            subdivisions: subdivisions.clone(),
            data_subdivisions: data_subdivisions.clone(),
            resolution_mapping,
        }
    }

    fn nodes(&self) -> &Vec<Self::Node> {
        &self.nodes
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
        // todo: this could be optimized by doing something like (trying to touch every node only once)
        //  - sorting the bricks by their level,
        //  - starting from the highest level, pushing nodes to process into a queue
        //  - processing this queue

        // todo: adapt to changes (i.e. bricks is a vechashmap now)
        // todo: this might need to map more nodes if multiple nodes map to the same level
        for b in bricks.get(&0).unwrap() {
            let normalized_address = b.global_address.index.to_normalized_address(
                self.data_subdivisions
                    .get(b.global_address.level as usize)
                    .unwrap(),
            );

            let level = self.map_to_subdivision_level(b.global_address.channel as usize);
            let subdivision = self.subdivisions.get(level).unwrap();
            let node_index = subdivision.to_node_index(normalized_address);
            self.nodes
                .get_mut(node_index)
                .unwrap()
                .set_mapped(b.min, b.max);

            if level > 0 {
                let mut child_node_index = node_index;
                for l in (0..level - 1).rev() {
                    let level_subdivision = self.subdivisions.get(l).unwrap();
                    let level_node_index = level_subdivision.to_node_index(normalized_address);
                    let subtree_index = 1 << (child_node_index - level_node_index * 8) as u8;
                    if !self
                        .nodes
                        .get_mut(level_node_index)
                        .unwrap()
                        .set_subtree_mapped(subtree_index)
                    {
                        break;
                    }
                    child_node_index = level_node_index;
                }
            }
        }
        todo!()
    }

    fn on_unmapped_bricks(&mut self, bricks: &VecHashMap<u32, UnmappedBrick>) {
        // todo: adapt to changes (i.e. bricks is a vechashmap now)
        // todo: this might need to unmap more nodes if multiple nodes map to the same level
        for b in bricks.get(&0).unwrap() {
            let normalized_address = b.global_address.index.to_normalized_address(
                self.data_subdivisions
                    .get(b.global_address.level as usize)
                    .unwrap(),
            );

            let level = self.map_to_subdivision_level(b.global_address.channel as usize);
            let subdivision = self.subdivisions.get(level).unwrap();
            let node_index = subdivision.to_node_index(normalized_address);
            let unmapped = self.nodes
                .get_mut(node_index)
                .unwrap()
                .set_unmapped();

            if unmapped && level > 0 {
                let mut child_node_index = node_index;
                for l in (0..level - 1).rev() {
                    let level_subdivision = self.subdivisions.get(l).unwrap();
                    let level_node_index = level_subdivision.to_node_index(normalized_address);
                    let subtree_index = 1 << (child_node_index - level_node_index * 8) as u8;
                    if !self
                        .nodes
                        .get_mut(level_node_index)
                        .unwrap()
                        .set_subtree_unmapped(subtree_index)
                    {
                        break;
                    }
                    child_node_index = level_node_index;
                }
            }
        }
        todo!()
    }
}
