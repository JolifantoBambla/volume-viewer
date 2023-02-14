use crate::app::renderer::dvr::page_table::PageTableDVR;
use crate::app::renderer::dvr::page_table_octree::PageTableOctreeDVR;
use crate::app::scene::volume::VolumeSceneObject;
use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupEntry, CommandEncoder, Extent3d};
use wgpu_framework::context::Gpu;
use wgsl_preprocessor::WGSLPreprocessor;

pub mod common;
pub mod page_table;
pub mod page_table_octree;

pub struct Resources<'a> {
    pub volume_sampler: &'a wgpu::Sampler,
    pub output: &'a wgpu::TextureView,
    pub uniforms: &'a wgpu::Buffer,
    pub channel_settings: &'a wgpu::Buffer,
}

impl<'a> AsBindGroupEntries for Resources<'a> {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(self.volume_sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(self.output),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.channel_settings.as_entire_binding(),
            },
        ]
    }
}

#[derive(Debug)]
pub enum RayGuidedDVR {
    PageTable(PageTableDVR),
    PageTableOctree(PageTableOctreeDVR),
}

impl RayGuidedDVR {
    pub fn new(
        volume: &VolumeSceneObject,
        wgsl_preprocessor: &WGSLPreprocessor,
        gpu: &Arc<Gpu>,
    ) -> Self {
        match volume {
            VolumeSceneObject::TopDownOctreeVolume(o) => {
                Self::PageTableOctree(PageTableOctreeDVR::new(
                    volume.volume_manager(),
                    o.octree(),
                    wgsl_preprocessor,
                    gpu,
                ))
            }
            VolumeSceneObject::PageTableVolume(_) => Self::PageTable(PageTableDVR::new(
                volume.volume_manager(),
                wgsl_preprocessor,
                gpu,
            )),
        }
    }

    pub fn create_bind_group(&self, resources: Resources) -> BindGroup {
        match self {
            RayGuidedDVR::PageTable(p) => p.create_bind_group(resources),
            RayGuidedDVR::PageTableOctree(o) => o.create_bind_group(resources),
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        output_extent: &Extent3d,
    ) {
        match self {
            RayGuidedDVR::PageTable(p) => p.encode(command_encoder, bind_group, output_extent),
            RayGuidedDVR::PageTableOctree(p) => {
                p.encode(command_encoder, bind_group, output_extent)
            }
        }
    }
}
