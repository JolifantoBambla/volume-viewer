use glam::UVec3;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{BufferAddress, Queue};

use crate::GPUContext;
use crate::resource::TypedBuffer;
use crate::volume::octree::subdivision::VolumeSubdivision;
use crate::volume::BrickedMultiResolutionMultiVolumeMeta;

pub mod direct_access_tree;
pub mod subdivision;
pub mod top_down_tree;

pub trait PageTableOctree {
    type Node: bytemuck::Pod;

    fn with_subdivisions(subdivisions: &Vec<VolumeSubdivision>) -> Self;

    fn nodes(&self) -> &Vec<Self::Node>;

    fn write_to_buffer(&self, buffer: TypedBuffer<Self::Node>, offset: u32, queue: &Queue) {
        queue.write_buffer(
            buffer.buffer(),
            offset as BufferAddress,
            bytemuck::cast_slice(self.nodes().as_slice())
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

    // todo: update
    //   - on new brick received
    //   - on channel selection change? -> no, each tree corresponds to one channel
}
