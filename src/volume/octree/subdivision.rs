use crate::util::extent::{SubscriptToIndex, ToSubscript};
use glam::{BVec3, UVec3, Vec3};
use std::ops::Range;

/// A helper struct describing a subdivision level in a full, and potentially pointerless octree.
/// It provides methods to compute indices into a list of octree nodes sorted by their subdivision
/// level from the lowest resolution to the highest resolution, i.e., the subdivision at index 0
/// will have one node only.
/// All indices computed by a `VolumeSubdivision`'s methods assume that there is only one channel.
/// That means that in case of a multi-channel octree, an index returned by a `VolumeSubdivision`'s
/// method must be translated to access the right node, e.g.:
/// For `n >= 1` channels, the actual index `i` in a list of interleaved nodes, i.e., a list that is
/// sorted by subdivision first and channel second, is: `i * n + c`, where `0 < c < n` is the
/// channel's index.
/// For `n >= 1` channels, the actual index `i` in a list of nodes that is sorted by channel first
/// and subdivision second is `c * m + i`, where `m` is the total number of nodes in a channel's
/// octree, and `0 < c < n` is the channel's index.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeSubdivision {
    /// The shape of this subdivision.
    shape: UVec3,

    /// This subdivision's offset into a list of nodes sorted by their subdivision from the lowest
    /// resolution to the highest resolution.
    /// Note that this does not take into account multiple channels.
    node_offset: u32,

    /// The shape of a node in terms of its child nodes in the next higher subdivision level.
    /// In a regular octree, i.e., where each node has 8 children, this would be `(2,2,2)`.
    children_shape: UVec3,

    /// The number of child nodes per node in this subdivision.
    /// In a regular octree, i.e., where each node has 8 children, this would be `8`.
    children_per_node: u32,
}

impl VolumeSubdivision {
    fn new(shape: UVec3, node_offset: u32) -> Self {
        Self {
            shape,
            node_offset,
            ..Default::default()
        }
    }

    pub fn from_input_and_target_shape(input_shape: UVec3, target_shape: UVec3) -> Vec<Self> {
        let mut subdivisions = Vec::new();
        let mut last_shape = input_shape;
        while last_shape.cmpgt(target_shape).any() {
            let subdivide = BVec3::new(
                last_shape.x > last_shape.y / 2 && last_shape.x > last_shape.z / 2,
                last_shape.y > last_shape.x / 2 && last_shape.y > last_shape.z / 2,
                last_shape.z > last_shape.x / 2 && last_shape.z > last_shape.y / 2,
            );
            last_shape = UVec3::new(
                if subdivide.x {
                    last_shape.x / 2
                } else {
                    last_shape.x
                },
                if subdivide.y {
                    last_shape.y / 2
                } else {
                    last_shape.y
                },
                if subdivide.z {
                    last_shape.z / 2
                } else {
                    last_shape.z
                },
            );
            subdivisions.push(subdivide);
        }
        subdivisions.reverse();

        let mut volume_subdivisions = vec![Self::default()];

        for (i, s) in subdivisions.iter().enumerate() {
            let children_shape = UVec3::new(
                if s.x { 2 } else { 1 },
                if s.y { 2 } else { 1 },
                if s.z { 2 } else { 1 },
            );
            volume_subdivisions
                .get_mut(i)
                .unwrap()
                .set_children_shape(children_shape);

            volume_subdivisions.push(Self::new(
                children_shape * volume_subdivisions.get(i).unwrap().shape,
                volume_subdivisions
                    .get(i)
                    .unwrap()
                    .next_subdivision_offset() as u32,
            ));
        }
        volume_subdivisions
    }

    fn set_children_shape(&mut self, children_shape: UVec3) {
        self.children_shape = children_shape;
        self.children_per_node = children_shape.x * children_shape.y * children_shape.z;
    }

    /// Returns the number of nodes in this subdivision.
    pub fn num_nodes(&self) -> usize {
        (self.shape.x * self.shape.y * self.shape.z) as usize
    }

    /// The index of the first node in this subdivision in a list of all octree nodes sorted by
    /// their subdivision.
    pub fn first_node_index(&self) -> u32 {
        self.node_offset
    }

    /// The index of the last node in this subdivision in a list of all octree nodes sorted by
    /// their subdivision.
    pub fn last_node_index(&self) -> usize {
        self.next_subdivision_offset() - 1
    }

    /// The offset of the first node in the next higher subdivision, i.e., the next higher
    /// resolution, in a list of all octree nodes sorted by their subdivision.
    pub fn next_subdivision_offset(&self) -> usize {
        self.node_offset as usize + self.num_nodes()
    }

    /// Computes the index of a node in this subdivision from a normalized address, i.e., a point in
    /// the unit cube ([0,1]^3), in a list of all octree nodes sorted by their subdivision.
    pub fn to_node_index(&self, normalized_address: Vec3) -> usize {
        (self.node_offset
            + normalized_address
                .to_subscript(self.shape())
                .to_index(&self.shape)) as usize
    }

    /// Computes a local node index into the slice corresponding to this `VolumeSubdivision` from a
    /// given global `node_index`, i.e., an index into a list of all nodes in the octree.
    pub fn local_node_index(&self, node_index: usize) -> usize {
        node_index - self.node_offset as usize
    }

    /// Computes the bit corresponding to a node's (`node_index`) subtree referenced by one of its
    /// child node's index (`child_node_index`).
    pub fn subtree_bit(&self, node_index: usize, child_node_index: usize) -> usize {
        child_node_index - self.first_child_node_index(node_index)
    }

    /// Computes a bit mask with only the bit of a node's (`node_index`) subtree corresponding to a
    /// child node's index (`child_node_index`) set.
    pub fn subtree_mask(&self, node_index: usize, child_node_index: usize) -> usize {
        1 << self.subtree_bit(node_index, child_node_index)
    }

    /// Computes a bitmask with all bits corresponding to a node's subtrees set.
    pub fn full_subtree_mask(&self) -> usize {
        let mut mask = 0;
        for i in 0..self.children_per_node {
            mask |= 1 << i;
        }
        mask as usize
    }

    /// Computes the index of a node's first child node from its own index (`node_index`).
    pub fn first_child_node_index(&self, node_index: usize) -> usize {
        self.next_subdivision_offset()
            + self.local_node_index(node_index) * self.children_per_node as usize
    }

    /// Computes the range of indices of a node's child nodes from its own index (`node_index`).
    pub fn child_node_indices(&self, node_index: usize) -> Range<usize> {
        let first_child_node_index = self.first_child_node_index(node_index);
        first_child_node_index..first_child_node_index + self.children_per_node as usize
    }

    pub fn shape(&self) -> UVec3 {
        self.shape
    }
}

impl Default for VolumeSubdivision {
    fn default() -> Self {
        Self {
            shape: UVec3::ONE,
            node_offset: 0,
            children_shape: UVec3::ONE,
            children_per_node: 1,
        }
    }
}

pub fn total_number_of_nodes(subdivisions: &[VolumeSubdivision]) -> usize {
    subdivisions.iter().fold(0, |acc, s| acc + s.num_nodes())
}
