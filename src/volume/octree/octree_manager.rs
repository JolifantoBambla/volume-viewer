use std::rc::Rc;
use std::sync::Arc;
use wgpu::BufferUsages;
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::Buffer;
use crate::volume::octree::MultiChannelPageTableOctreeDescriptor;
use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};

#[derive(Clone, Debug)]
pub struct OctreeManager {
    gpu: Arc<Gpu>,
    subdivisions: Vec<VolumeSubdivision>,
    max_num_channels: u32,
    gpu_subdivisions: Rc<Buffer<VolumeSubdivision>>,
    gpu_nodes: Rc<Buffer<u32>>,
}

impl OctreeManager {
    pub fn new(descriptor: MultiChannelPageTableOctreeDescriptor, gpu: &Arc<Gpu>) -> Self {
        let subdivisions = VolumeSubdivision::from_input_and_target_shape(
            descriptor.volume.resolutions[0].volume_size,
            descriptor.brick_size,
        );

        let gpu_subdivisions =
            Buffer::from_data("subdivisions", subdivisions.as_slice(), BufferUsages::STORAGE, gpu);

        let num_nodes_per_channel = total_number_of_nodes(subdivisions.as_slice());
        let initial_octree = vec![0; num_nodes_per_channel * descriptor.max_num_channels as usize];
        let gpu_buffer = Buffer::from_data(
            "octree",
            initial_octree.as_slice(),
            BufferUsages::STORAGE,
            gpu,
        );

        Self {
            gpu: gpu.clone(),
            subdivisions,
            max_num_channels: descriptor.max_num_channels,
            gpu_subdivisions: Rc::new(gpu_subdivisions),
            gpu_nodes: Rc::new(gpu_buffer),
        }
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }
    pub fn subdivisions(&self) -> &[VolumeSubdivision] {
        self.subdivisions.as_slice()
    }
    pub fn max_num_channels(&self) -> u32 {
        self.max_num_channels
    }
    pub fn gpu_subdivisions(&self) -> &Rc<Buffer<VolumeSubdivision>> {
        &self.gpu_subdivisions
    }
    pub fn gpu_nodes(&self) -> &Rc<Buffer<u32>> {
        &self.gpu_nodes
    }
    pub fn nodes_per_subdivision(&self) -> Vec<usize> {
        self.subdivisions.iter()
            .map(|s| s.num_nodes() * self.max_num_channels as usize)
            .collect()
    }
}
