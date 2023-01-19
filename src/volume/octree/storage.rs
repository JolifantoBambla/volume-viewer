struct InterleavedOctreeNodeStorage<'a, T> {
    num_channels: usize,
    channel_index: usize,
    nodes: &'a mut Vec<T>,
}

impl<'a, T> InterleavedOctreeNodeStorage<'a, T> {
    /// Gets a reference to a `Self::Node` by its index in this tree.
    /// If the tree is stored with other trees in an interleaved format, the given node index is
    /// translated to the node's actual index in the underlying data storage.
    pub fn node(&self, node_index: usize) -> Option<&T> {
        let index = node_index * self.num_channels + self.channel_index;
        self.nodes.get(index)
    }

    /// Gets a mutable reference to a `Self::Node` by its index in this tree.
    /// If the tree is stored with other trees in an interleaved format, the given node index is
    /// translated to the node's actual index in the underlying data storage.
    pub fn node_mut(&mut self, node_index: usize) -> Option<&mut T> {
        let index = node_index * self.num_channels + self.channel_index;
        self.nodes.get_mut(index)
    }
}

enum OctreeStorageType<'a, T> {
    Active(InterleavedOctreeNodeStorage<'a, T>),
    Inactive(&'a mut Vec<T>),
}

pub struct OctreeStorage<'a, T> {
    storage_type: OctreeStorageType<'a, T>,
}

impl<'a, T> OctreeStorage<'a, T> {
    pub fn new_active(num_channels: usize, channel_index: usize, nodes: &'a mut Vec<T>) -> Self {
        Self {
            storage_type: OctreeStorageType::Active(InterleavedOctreeNodeStorage {
                num_channels,
                channel_index,
                nodes,
            }),
        }
    }

    pub fn new_inactive(nodes: &'a mut Vec<T>) -> Self {
        Self {
            storage_type: OctreeStorageType::Inactive(nodes),
        }
    }

    /// Gets a reference to a `Self::Node` by its index in this tree.
    /// If the tree is stored with other trees in an interleaved format, the given node index is
    /// translated to the node's actual index in the underlying data storage.
    pub fn node(&self, node_index: usize) -> Option<&T> {
        match &self.storage_type {
            OctreeStorageType::Active(s) => s.node(node_index),
            OctreeStorageType::Inactive(s) => s.get(node_index),
        }
    }

    /// Gets a mutable reference to a `Self::Node` by its index in this tree.
    /// If the tree is stored with other trees in an interleaved format, the given node index is
    /// translated to the node's actual index in the underlying data storage.
    pub fn node_mut(&mut self, node_index: usize) -> Option<&mut T> {
        match &mut self.storage_type {
            OctreeStorageType::Active(s) => s.node_mut(node_index),
            OctreeStorageType::Inactive(s) => s.get_mut(node_index),
        }
    }
}
