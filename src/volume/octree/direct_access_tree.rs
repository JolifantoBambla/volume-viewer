use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};
use crate::volume::octree::{BrickCacheUpdateListener, MappedBrick, PageTableOctree, UnmappedBrick};

/*
#[modular_bitfield::bitfield]
#[repr(u8)]
#[derive(Copy, Clone, Debug, Default)]
struct MappedState {
    level_0_mapped: bool,
    level_1_mapped: bool,
    level_2_mapped: bool,
    level_3_mapped: bool,
    level_4_mapped: bool,
    level_5_mapped: bool,
    level_6_mapped: bool,
    level_7_mapped: bool,
}
 */

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    /// A bitfield that stores for each node on the path from this node to the root node including
    /// this node if it is mapped or not.
    /// Note that a node is also considered mapped if it is known to be empty or homogenous.
    mapped_levels: u8,

    /// The minimum value in the region in the volume represented by this node.
    min: u8,

    /// The maximum value in the region in the volume represented by this node.
    max: u8,

    /// The average value in the region in the volume represented by this node.
    average: u8,
}

#[derive(Clone, Debug)]
pub struct DirectAccessTree {
    #[allow(unused)]
    nodes: Vec<Node>,
}

impl DirectAccessTree {}

impl PageTableOctree for DirectAccessTree {
    type Node = Node;

    fn with_subdivisions(subdivisions: &[VolumeSubdivision]) -> Self {
        let nodes = vec![Node::default(); total_number_of_nodes(subdivisions) as usize];
        Self { nodes }
    }

    fn nodes(&self) -> &Vec<Self::Node> {
        &self.nodes
    }
}

impl BrickCacheUpdateListener for DirectAccessTree {
    fn on_mapped_bricks(&mut self, _bricks: &[MappedBrick]) {
        // todo: map brick address to node address
        // todo: handle empty bricks
        todo!()
    }

    fn on_unmapped_bricks(&mut self, _bricks: &[UnmappedBrick]) {
        todo!()
    }
}
