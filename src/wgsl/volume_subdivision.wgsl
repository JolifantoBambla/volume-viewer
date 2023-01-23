struct VolumeSubdivision {
    /// The shape of this subdivision.
    shape: vec3<u32>,

    /// This subdivision's offset into a list of nodes sorted by their subdivision from the lowest
    /// resolution to the highest resolution.
    /// Note that this does not take into account multiple channels.
    node_offset: u32,

    /// The shape of a node in terms of its child nodes in the next higher subdivision level.
    /// In a regular octree, i.e., where each node has 8 children, this would be `(2,2,2)`.
    children_shape: vec3<u32>,

    /// The number of child nodes per node in this subdivision.
    /// In a regular octree, i.e., where each node has 8 children, this would be `8`.
    children_per_node: u32,
}
