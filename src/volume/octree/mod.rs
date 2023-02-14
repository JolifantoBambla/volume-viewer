use crate::volume::BrickedMultiResolutionMultiVolumeMeta;
use glam::UVec3;

pub mod octree_manager;
pub mod subdivision;
pub mod update;

#[derive(Clone, Debug)]
pub struct OctreeDescriptor<'a> {
    pub volume: &'a BrickedMultiResolutionMultiVolumeMeta,
    pub leaf_node_size: UVec3,
    pub max_num_channels: usize,
}
