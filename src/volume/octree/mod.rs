use std::collections::HashMap;
use glam::{BVec3, UVec3, Vec3};
use crate::volume::BrickedMultiResolutionMultiVolumeMeta;
use crate::volume::octree::direct_access_tree::DirectAccessTree;
use crate::volume::octree::subdivision::VolumeSubdivision;
use crate::volume::octree::top_down_tree::TopDownTree;

pub mod direct_access_tree;
pub mod subdivision;
pub mod top_down_tree;

pub trait PageTableOctree {
    fn with_subdivision(subdivisions: &Vec<VolumeSubdivision>) -> Self;

    // todo: update
    //   - on new brick received
}

#[derive(Clone, Debug, Default)]
pub struct MultiChannelPageTableOctree<Tree: PageTableOctree> {
    subdivision: VolumeSubdivision,
    octrees: HashMap<usize, Tree>,
}

impl<Tree: PageTableOctree> MultiChannelPageTableOctree<Tree> {
    pub fn new(volume_descriptor: &BrickedMultiResolutionMultiVolumeMeta) -> Self {
        Self {
            subdivision: VolumeSubdivision::default(),
            octrees: HashMap::new()
        }
    }

    // todo: update
    //   - on new brick received
    //   - on channel selection change? -> no, each tree corresponds to one channel
}
