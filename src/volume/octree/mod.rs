use crate::renderer::settings::ChannelSettings;
use glam::UVec3;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::{BufferAddress, BufferUsages};
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::Buffer;

use crate::util::vec_hash_map::VecHashMap;
use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};
use crate::volume::{BrickAddress, BrickedMultiResolutionMultiVolumeMeta};

pub mod direct_access_tree;
pub mod subdivision;
pub mod top_down_tree;

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

#[readonly::make]
pub struct BrickCacheUpdateResult {
    mapped_bricks: HashMap<u32, VecHashMap<u32, MappedBrick>>,
    unmapped_bricks: HashMap<u32, VecHashMap<u32, UnmappedBrick>>,
}

impl BrickCacheUpdateResult {
    pub fn new(
        mapped_bricks: HashMap<u32, VecHashMap<u32, MappedBrick>>,
        unmapped_bricks: HashMap<u32, VecHashMap<u32, UnmappedBrick>>,
    ) -> Self {
        Self {
            mapped_bricks,
            unmapped_bricks,
        }
    }
}

pub trait BrickCacheUpdateListener {
    fn on_mapped_bricks(&mut self, bricks: &VecHashMap<u32, MappedBrick>);
    fn on_unmapped_bricks(&mut self, bricks: &VecHashMap<u32, UnmappedBrick>);
}

pub trait PageTableOctreeNode {
    fn from_resolution_mapping(resolution_mapping: usize) -> Self;
}

pub trait PageTableOctree: BrickCacheUpdateListener {
    type Node: bytemuck::Pod + Default + PageTableOctreeNode;

    fn create_nodes_from_subdivisions(
        subdivisions: &[VolumeSubdivision],
        resolution_mapping: &ResolutionMapping,
    ) -> Vec<Self::Node> {
        let mut nodes = Vec::with_capacity(total_number_of_nodes(subdivisions));
        for (octree_level, _) in subdivisions.iter().enumerate() {
            nodes.append(&mut vec![Self::Node::from_resolution_mapping(
                resolution_mapping.map_to_dataset_level(octree_level),
            )]);
        }
        nodes
    }

    fn new(
        subdivisions: &Rc<Vec<VolumeSubdivision>>,
        data_subdivisions: &Rc<Vec<UVec3>>,
        resolution_mapping: ResolutionMapping,
    ) -> Self;

    fn nodes(&self) -> &Vec<Self::Node>;

    fn subdivisions(&self) -> &Rc<Vec<VolumeSubdivision>>;

    fn resolution_mapping(&self) -> &ResolutionMapping;

    fn set_resolution_mapping(&mut self, resolution_mapping: ResolutionMapping);

    fn map_to_highest_subdivision_level(&self, level: usize) -> usize {
        *self
            .resolution_mapping()
            .map_to_octree_subdivision_level(level)
            .last()
            .unwrap_or(&0)
    }
}

#[derive(Clone, Debug)]
pub struct ResolutionMapping {
    /// The minimum (i.e., lowest) resolution level in the data set.
    min_resolution: u32,

    /// The maximum (i.e., highest) resolution level in the data set.
    max_resolution: u32,

    /// Maps resolution levels in the octree to their corresponding resolution levels in the data set.
    octree_to_dataset: Vec<usize>,

    /// Maps resolution levels in the data set to one ore more resolution levels in the octree.
    dataset_to_octree: VecHashMap<usize, usize>,
}

impl ResolutionMapping {
    pub fn new(
        octree_subdivisions: &[VolumeSubdivision],
        data_subdivisions: &[UVec3],
        min_lod: u32,
        max_lod: u32,
    ) -> Self {
        // subdivisions are ordered from low to high res
        // data_set_subdivisions are ordered from high res to low res
        // c.min_lod is lowest res for channel
        // c.max_lod is highest res for channel
        let mut octree_to_dataset = Vec::with_capacity(octree_subdivisions.len());
        let mut current_lod = min_lod;
        let mut reached_max_lod = current_lod == max_lod;
        for s in octree_subdivisions.iter() {
            // compare s and ds
            // if s <= ds: collect current_lod
            // else: collect next lod
            while !reached_max_lod
                && s.shape()
                    .cmpgt(*data_subdivisions.get(current_lod as usize).unwrap())
                    .any()
            {
                current_lod = max_lod.min(current_lod + 1);
                reached_max_lod = current_lod == max_lod;
            }
            octree_to_dataset.push(current_lod as usize);
        }
        let mut dataset_to_octree = VecHashMap::new();
        for (octree_index, dataset_index) in octree_to_dataset.iter().enumerate() {
            dataset_to_octree.insert(*dataset_index, octree_index);
        }

        Self {
            min_resolution: min_lod,
            max_resolution: max_lod,
            octree_to_dataset,
            dataset_to_octree,
        }
    }

    fn map_to_dataset_level(&self, octree_level: usize) -> usize {
        *self.octree_to_dataset.get(octree_level).unwrap()
    }

    fn map_to_octree_subdivision_level(&self, dataset_level: usize) -> &[usize] {
        self.dataset_to_octree
            .get(&dataset_level)
            .unwrap()
            .as_slice()
    }
}

impl PartialEq for ResolutionMapping {
    fn eq(&self, other: &Self) -> bool {
        self.min_resolution == other.min_resolution && self.max_resolution == other.max_resolution
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

    gpu_subdivisions: Rc<Buffer<VolumeSubdivision>>,
    gpu_nodes: Rc<Buffer<Tree::Node>>,
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

        let mut octrees = HashMap::new();
        for c in descriptor.visible_channels.iter() {
            // todo: pass channel lod mapping to tree constructor
            octrees.insert(
                *c,
                Tree::new(
                    &subdivisions,
                    &data_subdivisions,
                    resolution_mappings.remove(c).unwrap().clone(),
                ),
            );
        }

        let gpu_subdivisions =
            Buffer::from_data("subdivisions", &subdivisions, BufferUsages::STORAGE, gpu);

        // the buffer does not need to be initialized because all nodes are zero anyway
        let gpu_buffer = Buffer::new_zeroed(
            "octree",
            descriptor.max_num_channels as usize,
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
            gpu_subdivisions: Rc::new(gpu_subdivisions),
            gpu_nodes: Rc::new(gpu_buffer),
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
        for (channel, bricks) in update_result.mapped_bricks.iter() {
            self.assert_octree(*channel);
            self.octrees
                .get_mut(channel)
                .unwrap()
                .on_mapped_bricks(bricks);
        }
        for (channel, bricks) in update_result.unmapped_bricks.iter() {
            self.assert_octree(*channel);
            self.octrees
                .get_mut(channel)
                .unwrap()
                .on_unmapped_bricks(bricks);
        }

        // todo: update GPU buffers (either only those portions that changed or all of them)
        for (offset, channel) in self.visible_channels.iter().enumerate() {
            self.gpu_nodes.write_buffer_with_offset(
                self.octrees.get(channel).unwrap().nodes().as_slice(),
                offset as BufferAddress,
            );
        }
    }

    pub fn set_visible_channels(&mut self, visible_channels: &[u32]) {
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

    pub fn gpu_subdivisions(&self) -> &Rc<Buffer<VolumeSubdivision>> {
        &self.gpu_subdivisions
    }

    pub fn gpu_nodes(&self) -> &Rc<Buffer<Tree::Node>> {
        &self.gpu_nodes
    }
}
