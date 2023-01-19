use std::fmt::Debug;
use std::sync::Arc;
use glam::{Mat4, UVec3};
use wgpu_framework::context::Gpu;
use wgpu_framework::input::Input;
use crate::resource::VolumeManager;
use crate::volume::octree::page_table_octree::PageTableOctree;
use crate::volume::octree::top_down_tree::TopDownTree;
use crate::volume::octree::{MultiChannelPageTableOctree, MultiChannelPageTableOctreeDescriptor};

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

    pub fn new_octree_volume(descriptor: MultiChannelPageTableOctreeDescriptor, volume_manager: VolumeManager, gpu: &Arc<Gpu>) -> Self {
        Self::TopDownOctreeVolume(OctreeVolume {
            object_to_world: Self::make_volume_transform(&volume_manager),
            volume_manager,
            octree: MultiChannelPageTableOctree::new(descriptor, gpu),
        })
    }

    pub fn volume_transform(&self) -> Mat4 {
        match self {
            VolumeSceneObject::TopDownOctreeVolume(v) => v.object_to_world,
            VolumeSceneObject::PageTableVolume(v) => v.object_to_world
        }
    }

    pub fn volume_manager(&self) -> &VolumeManager {
        match self {
            VolumeSceneObject::TopDownOctreeVolume(v) => &v.volume_manager,
            VolumeSceneObject::PageTableVolume(v) => &v.volume_manager,
        }
    }

    pub fn volume_manager_mut(&mut self) -> &mut VolumeManager {
        match self {
            VolumeSceneObject::TopDownOctreeVolume(v) => &mut v.volume_manager,
            VolumeSceneObject::PageTableVolume(v) => &mut v.volume_manager,
        }
    }

    pub fn update_channel_selection(&mut self, visible_channels: Vec<u32>, timestamp: u32) -> Vec<u32> {
        let channel_mapping = self
            .volume_manager_mut()
            .add_channel_configuration(&visible_channels, timestamp)
            .iter()
            .map(|c| c.unwrap() as u32)
            .collect();

        if let VolumeSceneObject::TopDownOctreeVolume(v) = self {
            v.octree.set_visible_channels(visible_channels.as_slice());
        }

        channel_mapping
    }

    pub fn update_cache(&mut self, input: &Input) {
        let cache_update = self.volume_manager_mut().update_cache(input);
        if let VolumeSceneObject::TopDownOctreeVolume(v) = self {
            v.octree.on_brick_cache_updated(&cache_update);
        }
    }
}
