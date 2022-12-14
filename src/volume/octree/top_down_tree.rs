use crate::volume::octree::PageTableOctree;
use crate::volume::octree::subdivision::VolumeSubdivision;

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

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Node {
    /// A bitfield that stores for each of this node's subtrees this bitfield stores if it is
    /// partially mapped or not.
    /// Note that a subtree is also considered mapped if it is known to be empty or homogenous.
    partially_mapped_subtrees: u8,

    /// The minimum value in the region in the volume represented by this node.
    min: u8,

    /// The maximum value in the region in the volume represented by this node.
    max: u8,

    /// The average value in the region in the volume represented by this node.
    average: u8,
}

#[derive(Clone, Debug)]
pub struct TopDownTree {
    nodes: Vec<Node>,
}

impl PageTableOctree for TopDownTree {
    fn with_subdivision(subdivisions: &Vec<VolumeSubdivision>) -> Self {
        todo!()
    }
}
