use crate::renderer::settings::ChannelSettings;
use glam::UVec3;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::{BufferAddress, BufferUsages};
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::Buffer;

use crate::util::vec_hash_map::VecHashMap;
use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};
use crate::volume::{BrickAddress, BrickedMultiResolutionMultiVolumeMeta};
use crate::volume::octree::page_table_octree::PageTableOctree;
use crate::volume::octree::resolution_mapping::ResolutionMapping;
use crate::volume::octree::storage::OctreeStorage;

pub mod direct_access_tree;
pub mod subdivision;
pub mod top_down_tree;
pub mod page_table_octree;
pub mod resolution_mapping;
pub mod storage;

#[derive(Clone, Debug)]
pub struct MappedBrick {
    #[allow(unused)]
    global_address: BrickAddress,
    #[allow(unused)]
    min: u8,
    #[allow(unused)]
    max: u8,
}

impl MappedBrick {
    pub fn new(global_address: BrickAddress, min: u8, max: u8) -> Self {
        Self {
            global_address,
            min,
            max,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UnmappedBrick {
    #[allow(unused)]
    global_address: BrickAddress,
}

impl UnmappedBrick {
    pub fn new(global_address: BrickAddress) -> Self {
        Self { global_address }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChannelBrickCacheUpdateResult {
    mapped_bricks: VecHashMap<u32, MappedBrick>,
    unmapped_bricks: VecHashMap<u32, UnmappedBrick>,
}

impl ChannelBrickCacheUpdateResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_mapped(&mut self, brick: MappedBrick) {
        self.mapped_bricks.insert(
            brick.global_address.level,
            brick
        );
    }

    pub fn add_unmapped(&mut self, brick: UnmappedBrick) {
        self.unmapped_bricks.insert(
            brick.global_address.level,
            brick
        );
    }

    pub fn mapped_bricks(&self) -> &VecHashMap<u32, MappedBrick> {
        &self.mapped_bricks
    }

    pub fn unmapped_bricks(&self) -> &VecHashMap<u32, UnmappedBrick> {
        &self.unmapped_bricks
    }
}

#[readonly::make]
#[derive(Clone, Debug)]
pub struct BrickCacheUpdateResult(HashMap<u32, ChannelBrickCacheUpdateResult>);

impl BrickCacheUpdateResult {
    pub fn new(
        update_result: HashMap<u32, ChannelBrickCacheUpdateResult>,
    ) -> Self {
        Self {
            0: update_result,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiChannelPageTableOctreeDescriptor<'a> {
    pub volume: &'a BrickedMultiResolutionMultiVolumeMeta,
    pub channel_settings: &'a Vec<ChannelSettings>,

    /// The brick size of
    pub brick_size: UVec3,

    /// The maximum number of channels that can be represented on the GPU.
    /// Whether this number of channels is used or not, this determines the GPU memory that is
    /// allocated for this octree.
    pub max_num_channels: u32,

    /// A sorted list of visible channels.
    pub visible_channels: Vec<u32>,

    pub interleaved_storage: bool,
}

#[derive(Clone, Debug)]
pub struct MultiChannelPageTableOctree<Tree: PageTableOctree> {
    gpu: Arc<Gpu>,

    subdivisions: Rc<Vec<VolumeSubdivision>>,

    data_subdivisions: Rc<Vec<UVec3>>,

    resolution_mappings: HashMap<u32, ResolutionMapping>,

    octrees: HashMap<u32, Tree>,

    /// A sorted list of visible channels.
    visible_channels: Vec<u32>,
    max_num_channels: u32,

    gpu_subdivisions: Rc<Buffer<VolumeSubdivision>>,
    gpu_nodes: Rc<Buffer<Tree::Node>>,

    active_node_storage: Vec<Tree::Node>,
    inactive_node_storage: HashMap<u32, Vec<Tree::Node>>,
}

impl<Tree: PageTableOctree> MultiChannelPageTableOctree<Tree> {
    pub fn new(descriptor: MultiChannelPageTableOctreeDescriptor, gpu: &Arc<Gpu>) -> Self {
        let subdivisions = Rc::new(VolumeSubdivision::from_input_and_target_shape(
            descriptor.volume.resolutions[0].volume_size,
            descriptor.brick_size,
        ));

        // sorted from lowest to highest, just like subdivisions
        let data_set_subdivisions: Vec<UVec3> = descriptor
            .volume
            .resolutions
            .iter()
            .map(|r| r.volume_size)
            .collect();

        let mut resolution_mappings = HashMap::new();
        for c in descriptor.channel_settings.iter() {
            resolution_mappings.insert(
                c.channel_index,
                ResolutionMapping::new(
                    subdivisions.as_slice(),
                    data_set_subdivisions.as_slice(),
                    c.min_lod,
                    c.max_lod,
                ),
            );
        }

        let data_subdivisions = Rc::new(data_set_subdivisions);

        let num_nodes_per_channel = total_number_of_nodes(subdivisions.as_slice());
        let mut active_node_storage = vec![Tree::Node::default(); num_nodes_per_channel * descriptor.max_num_channels as usize];

        let mut octrees = HashMap::new();
        for (channel_index, c) in descriptor.visible_channels.iter().enumerate() {
            // todo: pass channel lod mapping to tree constructor
            octrees.insert(
                *c,
                Tree::new(
                    &subdivisions,
                    &data_subdivisions,
                    resolution_mappings.remove(c).unwrap().clone(),
                ),
            );

            octrees.get(c)
                .unwrap()
                .create_nodes()
                .drain(..)
                .enumerate()
                .for_each(|(local_node_index, node)| {
                    let global_node_index = if descriptor.interleaved_storage {
                        local_node_index * descriptor.max_num_channels as usize + channel_index
                    } else {
                        local_node_index + num_nodes_per_channel * descriptor.max_num_channels as usize
                    };
                    active_node_storage[global_node_index] = node;
                });
        }

        let gpu_subdivisions =
            Buffer::from_data("subdivisions", &subdivisions, BufferUsages::STORAGE, gpu);

        // the buffer does not need to be initialized because all nodes are zero anyway
        let gpu_buffer = Buffer::from_data(
            "octree",
            &active_node_storage,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            gpu,
        );

        Self {
            gpu: gpu.clone(),
            subdivisions,
            data_subdivisions,
            octrees,
            resolution_mappings,
            visible_channels: descriptor.visible_channels,
            max_num_channels: descriptor.max_num_channels,
            gpu_subdivisions: Rc::new(gpu_subdivisions),
            gpu_nodes: Rc::new(gpu_buffer),
            active_node_storage,
            inactive_node_storage: HashMap::new(),
        }
    }

    pub fn on_resolution_mapping_updated(&mut self, channel: u32, min_lod: u32, max_lod: u32) {
        let resolution_mapping = ResolutionMapping::new(
            self.subdivisions.as_slice(),
            self.data_subdivisions.as_slice(),
            min_lod,
            max_lod,
        );
        self.assert_octree(channel);
        self.octrees
            .get_mut(&channel)
            .unwrap()
            .set_resolution_mapping(resolution_mapping);
    }

    pub fn assert_octree(&mut self, channel: u32) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.octrees.entry(channel) {
            e.insert(Tree::new(
                &self.subdivisions,
                &self.data_subdivisions,
                self.resolution_mappings.remove(&channel).unwrap(),
            ));
        }
    }

    pub fn on_brick_cache_updated(&mut self, update_result: &BrickCacheUpdateResult) {
        for (channel, update) in update_result.0.iter() {
            self.assert_octree(*channel);
            let mut node_storage = if self.visible_channels.contains(channel) {
                OctreeStorage::new_active(
                    self.max_num_channels as usize,
                    *channel as usize,
                    &mut self.active_node_storage
                )
            } else {
                OctreeStorage::new_inactive(self.inactive_node_storage.get_mut(channel).unwrap())
            };
            self.octrees
                .get_mut(channel)
                .unwrap()
                .on_brick_cache_update(update, &mut node_storage);
        }

        /*
        // todo: update GPU buffers (either only those portions that changed or all of them)
        for (offset, channel) in self.visible_channels.iter().enumerate() {
            self.gpu_nodes.write_buffer_with_offset(
                self.octrees.get(channel).unwrap().nodes().as_slice(),
                offset as BufferAddress,
            );
        }

         */
    }

    pub fn set_visible_channels(&mut self, visible_channels: &[u32]) {
        // todo: I kinda decided that the multichannel octree would use the same channel ordering as the page table to keep access consistent and don't introduce an extra indirection
        self.visible_channels = visible_channels.to_owned();
        for &c in self.visible_channels.iter() {
            if let std::collections::hash_map::Entry::Vacant(e) = self.octrees.entry(c) {
                e.insert(Tree::new(
                    &self.subdivisions,
                    &self.data_subdivisions,
                    self.resolution_mappings.remove(&c).unwrap(),
                ));
            }
        }
    }

    // todo: handle changes to resolution mapping

    pub fn gpu_subdivisions(&self) -> &Rc<Buffer<VolumeSubdivision>> {
        &self.gpu_subdivisions
    }

    pub fn gpu_nodes(&self) -> &Rc<Buffer<Tree::Node>> {
        &self.gpu_nodes
    }
}
