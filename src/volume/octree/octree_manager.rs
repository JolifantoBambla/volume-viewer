use crate::volume::octree::subdivision::{total_number_of_nodes, VolumeSubdivision};
use crate::volume::octree::OctreeDescriptor;
use std::cmp::min;
use std::sync::Arc;
use wgpu::{BindingResource, BufferUsages};
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::Buffer;

// the default node is min=255, max=0 -> default_node = 255;
const DEFAULT_NODE: u32 = 255;

#[derive(Debug)]
pub struct Octree {
    gpu: Arc<Gpu>,
    subdivisions: Vec<VolumeSubdivision>,
    max_num_channels: usize,
    gpu_subdivisions: Buffer<VolumeSubdivision>,
    gpu_nodes: Buffer<u32>,
}

impl Octree {
    pub fn new(descriptor: OctreeDescriptor, gpu: &Arc<Gpu>) -> Self {
        let max_num_channels = min(
            descriptor.max_num_channels,
            descriptor.volume.channels.len(),
        );

        // todo: this should use the volume size adjusted by voxel spacing instead
        let subdivisions = VolumeSubdivision::from_input_and_target_shape(
            descriptor.volume.resolutions[0].volume_size,
            descriptor.leaf_node_size,
        );

        log::info!("octree subdivisions: {:?}", subdivisions);

        let gpu_subdivisions = Buffer::from_data(
            "subdivisions",
            subdivisions.as_slice(),
            BufferUsages::STORAGE,
            gpu,
        );

        let num_nodes_per_channel = total_number_of_nodes(subdivisions.as_slice());
        let initial_octree = vec![DEFAULT_NODE; num_nodes_per_channel * max_num_channels];
        let gpu_buffer = Buffer::from_data(
            "octree",
            initial_octree.as_slice(),
            BufferUsages::STORAGE,
            gpu,
        );

        Self {
            gpu: gpu.clone(),
            subdivisions,
            max_num_channels,
            gpu_subdivisions,
            gpu_nodes: gpu_buffer,
        }
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }
    pub fn subdivisions(&self) -> &[VolumeSubdivision] {
        self.subdivisions.as_slice()
    }
    pub fn max_num_channels(&self) -> usize {
        self.max_num_channels
    }
    pub fn nodes_per_subdivision(&self) -> Vec<usize> {
        self.subdivisions
            .iter()
            .map(|s| s.num_nodes() * self.max_num_channels())
            .collect()
    }
    pub fn octree_nodes_as_binding_resource(&self) -> BindingResource {
        self.gpu_nodes.buffer().as_entire_binding()
    }
    pub fn volume_subdivisions_as_binding_resource(&self) -> BindingResource {
        self.gpu_subdivisions.buffer().as_entire_binding()
    }
}
