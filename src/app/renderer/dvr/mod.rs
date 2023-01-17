use wgpu::{BindGroup, CommandEncoder, Extent3d};
use crate::app::renderer::dvr::page_table::PageTableDVR;
use crate::app::renderer::dvr::page_table_octree::PageTableOctreeDVR;

pub mod common;
pub mod page_table;
pub mod page_table_octree;

#[derive(Debug)]
pub enum RayGuidedDVR {
    PageTable(PageTableDVR),
    PageTableOctree(PageTableOctreeDVR),
}

impl RayGuidedDVR {
    pub fn encode(&self, command_encoder: &mut CommandEncoder, bind_group: &BindGroup, output_extent: &Extent3d) {
        match self {
            RayGuidedDVR::PageTable(p) => p.encode(command_encoder, bind_group, output_extent),
            RayGuidedDVR::PageTableOctree(p) => p.encode(command_encoder, bind_group, output_extent),
        }
    }
}
