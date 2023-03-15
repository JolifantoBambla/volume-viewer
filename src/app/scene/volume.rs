use crate::resource::VolumeManager;
#[cfg(feature = "timestamp-query")]
use crate::timing::timestamp_query_helper::TimestampQueryHelper;
use crate::volume::octree::octree_manager::Octree;
use crate::volume::octree::update::OctreeUpdate;
use crate::volume::octree::OctreeDescriptor;
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

impl OctreeVolume {
    pub fn octree(&self) -> &Octree {
        &self.octree
    }
}

#[derive(Debug)]
pub struct PageTableVolume {
    object_to_world: Mat4,
    volume_manager: VolumeManager,
}

#[derive(Debug)]
pub enum VolumeSceneObject {
    TopDownOctreeVolume(Box<OctreeVolume>),
    PageTableVolume(Box<PageTableVolume>),
}

impl VolumeSceneObject {
    fn make_volume_transform(volume_manager: &VolumeManager) -> Mat4 {
        log::info!("normalized volume size: {:?}", volume_manager.normalized_volume_size());
        Mat4::from_scale(volume_manager.normalized_volume_size()).mul_mat4(
            &glam::Mat4::from_translation(glam::Vec3::new(-0.5, -0.5, -0.5)),
        )
    }

    pub fn new_page_table_volume(volume_manager: VolumeManager) -> Self {
        Self::PageTableVolume(Box::new(PageTableVolume {
            object_to_world: Self::make_volume_transform(&volume_manager),
            volume_manager,
        }))
    }

    pub fn new_octree_volume(
        descriptor: OctreeDescriptor,
        volume_manager: VolumeManager,
        wgsl_preprocessor: &WGSLPreprocessor,
        gpu: &Arc<Gpu>,
    ) -> Self {
        let octree = Octree::new(descriptor, gpu);
        let octree_update = OctreeUpdate::new(&octree, &volume_manager, wgsl_preprocessor);
        Self::TopDownOctreeVolume(Box::new(OctreeVolume {
            object_to_world: Self::make_volume_transform(&volume_manager),
            volume_manager,
            octree,
            octree_update,
        }))
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
        visible_channels: &Vec<u32>,
        timestamp: u32,
    ) -> Vec<u32> {
        let channel_mapping = self
            .volume_manager_mut()
            .add_channel_configuration(visible_channels, timestamp)
            .iter()
            .map(|c| c.unwrap() as u32)
            .collect();

        if let VolumeSceneObject::TopDownOctreeVolume(_v) = self {
            // todo: handle new channel selection in octree update
            //v.octree.set_visible_channels(visible_channels.as_slice());
        }

        channel_mapping
    }

    pub fn update_cache(
        &mut self,
        input: &Input,
        #[cfg(feature = "timestamp-query")] timestamp_query_helper: &mut TimestampQueryHelper,
    ) {
        let cache_update = self.volume_manager_mut().update_cache(input);
        if let VolumeSceneObject::TopDownOctreeVolume(v) = self {
            // try reading update's frame's buffer
            #[cfg(feature = "timestamp-query")]
            timestamp_query_helper.read_buffer();

            if !cache_update.is_empty() {
                let gpu = v.octree.gpu();
                let mut command_encoder =
                    gpu.device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Label::from("octree update"),
                        });

                v.octree_update.on_brick_cache_updated(
                    &mut command_encoder,
                    &cache_update,
                    #[cfg(feature = "timestamp-query")]
                    timestamp_query_helper,
                );

                v.octree_update.copy_to_readable(&mut command_encoder);

                #[cfg(feature = "timestamp-query")]
                timestamp_query_helper.resolve(&mut command_encoder);

                gpu.queue().submit(Some(command_encoder.finish()));

                // map this update's buffer and prepare next update
                #[cfg(feature = "timestamp-query")]
                timestamp_query_helper.map_buffer();

                if cache_update.mapped_local_brick_ids().len() >= 1 {
                    v.octree_update.map_break_point();
                }
            }
            v.octree_update.maybe_print_break_point();
        }
    }
}
