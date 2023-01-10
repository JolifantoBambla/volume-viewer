use glam::UVec3;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{BufferAddress, Queue};

use crate::resource::TypedBuffer;
use crate::volume::octree::subdivision::VolumeSubdivision;
use crate::volume::{Brick, BrickAddress, BrickedMultiResolutionMultiVolumeMeta};
use crate::GPUContext;

pub mod direct_access_tree;
pub mod subdivision;
pub mod top_down_tree;

#[derive(Clone, Debug)]
pub struct MappedBrick {
    global_address: BrickAddress,
    local_address: BrickAddress,
    brick: Brick,
}

impl MappedBrick {
    pub fn new(global_address: BrickAddress, local_address: BrickAddress, brick: Brick) -> Self {
        Self {
            global_address,
            local_address,
            brick,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UnmappedBrick {
    global_address: BrickAddress,
    local_address: BrickAddress,
}

impl UnmappedBrick {
    pub fn new(global_address: BrickAddress, local_address: BrickAddress) -> Self {
        Self {
            global_address,
            local_address,
        }
    }
}

pub trait BrickCacheUpdateListener {
    fn on_mapped_bricks(&mut self, bricks: &[MappedBrick]);
    fn on_unmapped_bricks(&mut self, bricks: &[UnmappedBrick]);
}

pub trait PageTableOctree {
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

    // todo: update
    //   - on new brick received
}

#[derive(Clone, Debug)]
pub struct MultiChannelPageTableOctreeDescriptor<'a> {
    pub volume: &'a BrickedMultiResolutionMultiVolumeMeta,
    pub brick_size: UVec3,
    pub num_channels: u32,
}

#[derive(Debug)]
struct GpuData {}

#[derive(Clone, Debug)]
pub struct MultiChannelPageTableOctree<Tree: PageTableOctree> {
    #[allow(unused)]
    gpu: Arc<GPUContext>,
    #[allow(unused)]
    subdivisions: Vec<VolumeSubdivision>,
    #[allow(unused)]
    octrees: HashMap<usize, Tree>,
}

impl<Tree: PageTableOctree> MultiChannelPageTableOctree<Tree> {
    pub fn new(descriptor: MultiChannelPageTableOctreeDescriptor, gpu: &Arc<GPUContext>) -> Self {
        let subdivisions = VolumeSubdivision::from_input_and_target_shape(
            descriptor.volume.resolutions[0].volume_size,
            descriptor.brick_size,
        );

        Self {
            gpu: gpu.clone(),
            subdivisions,
            octrees: HashMap::new(),
        }
    }
}

impl<Tree: PageTableOctree> BrickCacheUpdateListener for MultiChannelPageTableOctree<Tree> {
    fn on_mapped_bricks(&mut self, bricks: &[MappedBrick]) {
        todo!("update tree")
    }

    fn on_unmapped_bricks(&mut self, bricks: &[UnmappedBrick]) {
        todo!("update tree")
    }
}
