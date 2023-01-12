use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};
use crate::volume::octree::{
    BrickCacheUpdateListener, MappedBrick, PageTableOctree, ResolutionMapping, UnmappedBrick,
};
use js_sys::Atomics::sub;
use std::rc::Rc;

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

#[derive(Clone, Debug)]
pub struct TopDownTree {
    nodes: Vec<Node>,
    subdivisions: Rc<Vec<VolumeSubdivision>>,
    resolution_mapping: ResolutionMapping,
}

impl PageTableOctree for TopDownTree {
    type Node = Node;

    fn new(
        subdivisions: &Rc<Vec<VolumeSubdivision>>,
        resolution_mapping: ResolutionMapping,
    ) -> Self {
        Self {
            nodes: Self::create_nodes_from_subdivisions(subdivisions.as_slice()),
            subdivisions: subdivisions.clone(),
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
    fn on_mapped_bricks(&mut self, bricks: &[MappedBrick]) {
        // todo: handle empty bricks
        todo!()
    }

    fn on_unmapped_bricks(&mut self, bricks: &[UnmappedBrick]) {
        todo!()
    }
}
