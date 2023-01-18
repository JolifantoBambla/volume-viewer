use crate::resource::VolumeManager;
use crate::volume::octree::page_table_octree::PageTableOctree;
use crate::volume::octree::top_down_tree::TopDownTree;
use crate::volume::octree::MultiChannelPageTableOctree;
use glam::Mat4;
use std::fmt::Debug;

#[derive(Debug)]
pub struct OctreeVolume<T: PageTableOctree> {
    object_to_world: Mat4,
    octree: MultiChannelPageTableOctree<T>,
    volume_manager: VolumeManager,
}

#[derive(Debug)]
pub struct PageTableVolume {
    object_to_world: Mat4,
    volume_manager: VolumeManager,
}

#[derive(Debug)]
pub enum VolumeSceneObject {
    TopDownOctreeVolume(OctreeVolume<TopDownTree>),
    PageTableVolume(PageTableVolume),
}

impl VolumeSceneObject {
    fn make_volume_transform(volume_manager: &VolumeManager) -> Mat4 {
        Mat4::from_scale(volume_manager.normalized_volume_size()).mul_mat4(
            &glam::Mat4::from_translation(glam::Vec3::new(-0.5, -0.5, -0.5)),
        )
    }

    pub fn new_page_table_volume(volume_manager: VolumeManager) -> Self {
        Self::PageTableVolume(PageTableVolume {
            object_to_world: Self::make_volume_transform(&volume_manager),
            volume_manager,
        })
    }

    //pub fn new_octree_volume(volume_manager: VolumeManager) -> Self {}
}
