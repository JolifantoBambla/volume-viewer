use crate::resource::VolumeManager;
use crate::volume::octree::octree_manager::Octree;
use crate::volume::octree::update::{CacheUpdateMeta, OctreeUpdate};
use crate::volume::octree::MultiChannelPageTableOctreeDescriptor;
use glam::Mat4;
use std::fmt::Debug;
use std::sync::Arc;
use wgpu::Label;
use wgpu_framework::context::Gpu;
use wgpu_framework::input::Input;
use wgsl_preprocessor::WGSLPreprocessor;

#[derive(Debug)]
pub struct OctreeVolume {
    object_to_world: Mat4,
    octree: Octree,
    octree_update: OctreeUpdate,
    volume_manager: VolumeManager,
}

#[derive(Debug)]
pub struct PageTableVolume {
    object_to_world: Mat4,
    volume_manager: VolumeManager,
}

#[derive(Debug)]
pub enum VolumeSceneObject {
    TopDownOctreeVolume(OctreeVolume),
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

    pub fn new_octree_volume(
        descriptor: MultiChannelPageTableOctreeDescriptor,
        volume_manager: VolumeManager,
        wgsl_preprocessor: &WGSLPreprocessor,
        gpu: &Arc<Gpu>,
    ) -> Self {
        let octree = Octree::new(descriptor, gpu);
        let octree_update = OctreeUpdate::new(&octree, &volume_manager, wgsl_preprocessor);
        Self::TopDownOctreeVolume(OctreeVolume {
            object_to_world: Self::make_volume_transform(&volume_manager),
            volume_manager,
            octree,
            octree_update,
        })
    }

    pub fn volume_transform(&self) -> Mat4 {
        match self {
            VolumeSceneObject::TopDownOctreeVolume(v) => v.object_to_world,
            VolumeSceneObject::PageTableVolume(v) => v.object_to_world,
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

    pub fn update_channel_selection(
        &mut self,
        visible_channels: Vec<u32>,
        timestamp: u32,
    ) -> Vec<u32> {
        let channel_mapping = self
            .volume_manager_mut()
            .add_channel_configuration(&visible_channels, timestamp)
            .iter()
            .map(|c| c.unwrap() as u32)
            .collect();

        if let VolumeSceneObject::TopDownOctreeVolume(_v) = self {
            // todo: handle new channel selection in octree update
            //v.octree.set_visible_channels(visible_channels.as_slice());
        }

        channel_mapping
    }

    pub fn update_cache(&mut self, input: &Input) {
        let _cache_update = self.volume_manager_mut().update_cache(input);
        if let VolumeSceneObject::TopDownOctreeVolume(v) = self {
            let gpu = v.octree.gpu();
            let mut command_encoder =
                gpu.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Label::from("octree update"),
                    });

            let cache_update_meta = CacheUpdateMeta::default();
            v.octree_update
                .on_brick_cache_updated(&mut command_encoder, &cache_update_meta);

            gpu.queue().submit(Some(command_encoder.finish()));
        }
    }
}
