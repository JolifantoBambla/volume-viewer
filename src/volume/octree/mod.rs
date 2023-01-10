use glam::UVec3;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{BufferAddress, Queue};
use wgpu_framework::context::Gpu;

use crate::resource::TypedBuffer;
use crate::volume::octree::subdivision::VolumeSubdivision;
use crate::volume::{BrickAddress, BrickedMultiResolutionMultiVolumeMeta};
use crate::util::vec_hash_map::VecHashMap;

pub mod direct_access_tree;
pub mod subdivision;
pub mod top_down_tree;

#[derive(Clone, Debug)]
pub struct MappedBrick {
    global_address: BrickAddress,
    min: u8,
    max: u8,
}

impl MappedBrick {
    pub fn new(global_address: BrickAddress, min: u8, max: u8) -> Self {
        Self { global_address, min, max }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UnmappedBrick {
    global_address: BrickAddress,
}

impl UnmappedBrick {
    pub fn new(global_address: BrickAddress) -> Self {
        Self {
            global_address,
        }
    }
}

#[readonly::make]
pub struct BrickCacheUpdateResult {
    mapped_bricks: VecHashMap<u32, MappedBrick>,
    unmapped_bricks: VecHashMap<u32, UnmappedBrick>,
}

impl BrickCacheUpdateResult {
    pub fn new(mapped_bricks: VecHashMap<u32, MappedBrick>, unmapped_bricks: VecHashMap<u32, UnmappedBrick>) -> Self {
        Self { mapped_bricks, unmapped_bricks }
    }
}

pub trait BrickCacheUpdateListener {
    fn on_mapped_bricks(&mut self, bricks: &[MappedBrick]);
    fn on_unmapped_bricks(&mut self, bricks: &[UnmappedBrick]);
}

pub trait PageTableOctree: BrickCacheUpdateListener {
    type Node: bytemuck::Pod;

    fn with_subdivisions(subdivisions: &[VolumeSubdivision]) -> Self;

    fn nodes(&self) -> &Vec<Self::Node>;

    fn write_to_buffer(&self, buffer: TypedBuffer<Self::Node>, offset: u32, queue: &Queue) {
        queue.write_buffer(
            buffer.buffer(),
            offset as BufferAddress,
            bytemuck::cast_slice(self.nodes().as_slice()),
        );
    }
}

#[derive(Clone, Debug)]
pub struct MultiChannelPageTableOctreeDescriptor<'a> {
    pub volume: &'a BrickedMultiResolutionMultiVolumeMeta,
    pub brick_size: UVec3,
    pub num_channels: u32,
    pub visible_channels: Vec<u32>,
}

#[derive(Debug)]
struct GpuData {}

#[derive(Clone, Debug)]
pub struct MultiChannelPageTableOctree<Tree: PageTableOctree> {
    #[allow(unused)]
    gpu: Arc<Gpu>,
    subdivisions: Vec<VolumeSubdivision>,
    octrees: HashMap<u32, Tree>,
    visible_channels: Vec<u32>,
}

impl<Tree: PageTableOctree> MultiChannelPageTableOctree<Tree> {
    pub fn new(descriptor: MultiChannelPageTableOctreeDescriptor, gpu: &Arc<Gpu>) -> Self {
        let subdivisions = VolumeSubdivision::from_input_and_target_shape(
            descriptor.volume.resolutions[0].volume_size,
            descriptor.brick_size,
        );

        let mut octrees = HashMap::new();
        for &c in descriptor.visible_channels.iter() {
            octrees.insert(c, Tree::with_subdivisions(subdivisions.as_slice()));
        }

        Self {
            gpu: gpu.clone(),
            subdivisions,
            octrees,
            visible_channels: descriptor.visible_channels
        }
    }

    pub fn on_brick_cache_updated(&mut self, update_result: &BrickCacheUpdateResult) {
        for (channel, bricks) in update_result.mapped_bricks.iter() {
            if !self.octrees.contains_key(channel) {
                self.octrees.insert(*channel, Tree::with_subdivisions(self.subdivisions.as_slice()));
            }
            self.octrees.get_mut(channel)
                .unwrap()
                .on_mapped_bricks(bricks.as_slice());
        }
        for (channel, bricks) in update_result.unmapped_bricks.iter() {
            if !self.octrees.contains_key(channel) {
                self.octrees.insert(*channel, Tree::with_subdivisions(self.subdivisions.as_slice()));
            }
            self.octrees.get_mut(channel)
                .unwrap()
                .on_unmapped_bricks(bricks.as_slice());
        }

        // todo: update GPU buffers (either only those portions that changed or all of them)
        for channel in self.visible_channels.iter() {
            /*
                self.octrees.get(c)
                    .unwrap()
                    .write_to_buffer();
            */
        }
    }

    pub fn set_visible_channels(&mut self, visible_channels: &[u32]) {
        self.visible_channels = visible_channels.to_owned();
        for &c in self.visible_channels.iter() {
            if let std::collections::hash_map::Entry::Vacant(e) = self.octrees.entry(c) {
                e.insert(Tree::with_subdivisions(self.subdivisions.as_slice()));
            }
        }
    }
}
