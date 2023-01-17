use crate::util::extent::SubscriptToIndex;
use crate::volume::{BrickedMultiResolutionMultiVolumeMeta, ResolutionMeta};
use glam::{UVec2, UVec3, Vec3};
use std::cmp::min;

#[repr(C)]
#[derive(Clone, Debug)]
pub struct PageTableMeta {
    /// The offset of this page table in the page directory.
    offset: UVec3,

    /// The extent of this page table in the page directory.
    /// The extent of the full volume represented by this page table is the component-wise product
    /// `self.extent * brick_size`, where `brick_size` is the size of one page in cache.
    extent: UVec3,

    ///
    resolution_meta: ResolutionMeta,
}

impl PageTableMeta {
    pub fn new(offset: UVec3, extent: UVec3, volume_meta: ResolutionMeta) -> Self {
        Self {
            offset,
            extent,
            resolution_meta: volume_meta,
        }
    }

    pub fn get_max_location(&self) -> UVec3 {
        self.offset + self.extent
    }

    pub fn offset(&self) -> UVec3 {
        self.offset
    }

    pub fn extent(&self) -> UVec3 {
        self.extent
    }

    pub fn resolution_meta(&self) -> &ResolutionMeta {
        &self.resolution_meta
    }
}

#[derive(Clone, Debug)]
pub struct PageDirectoryMeta {
    /// The size of a brick in the brick cache. This is constant across all channels and resolutions
    /// of the bricked multi-resolution volume.
    brick_size: UVec3,

    normalized_volume_size: Vec3,

    extent: UVec3,

    /// The resolutions
    page_tables: Vec<PageTableMeta>,

    num_channels: usize,
    num_resolutions: usize,
}

impl PageDirectoryMeta {
    // todo: configure how many channels the page table can hold
    // todo: more efficient packing strategy
    pub fn new(
        volume_meta: &BrickedMultiResolutionMultiVolumeMeta,
        max_channels: usize,
        max_resolutions: usize,
    ) -> Self {
        let num_channels = min(max_channels, volume_meta.channels.len());
        let num_resolutions = min(max_resolutions, volume_meta.resolutions.len());

        let mut page_tables: Vec<PageTableMeta> = Vec::new();
        let high_res_extent = volume_meta.bricks_per_dimension(0);

        let packing_axis = if high_res_extent.x == high_res_extent.min_element() {
            UVec3::X
        } else if high_res_extent.y == high_res_extent.min_element() {
            UVec3::Y
        } else {
            UVec3::Z
        };
        let mut last_offset = UVec3::ZERO;
        let mut last_extent = UVec3::ZERO;
        for (level, volume_resolution) in volume_meta.resolutions.iter().enumerate() {
            // todo: currently, all channels share the same resolution. however, it might be that
            //   channel a should be present in resolutions 1-6 and channel b in resolutions 4-10,
            //   and only 6 resolutions can be represented at once.
            //   this construction still works if resolutions are strictly decreasing all dimensions
            //   which is typically a safe assumption to make. then only the extent has to be
            //   updated for rendering
            if level < num_resolutions {
                let extent = volume_meta.bricks_per_dimension(level);
                for _ in 0..num_channels {
                    let offset = last_offset + last_extent * packing_axis;
                    page_tables.push(PageTableMeta::new(
                        offset,
                        extent,
                        volume_resolution.clone(),
                    ));
                    last_offset = offset;
                    last_extent = extent;
                }
            }
        }

        let extent = page_tables
            .iter()
            .fold(UVec3::ZERO, |a, b| a.max(b.get_max_location()));

        Self {
            brick_size: volume_meta.brick_size,
            extent,
            normalized_volume_size: volume_meta.top_level_normalized_size(),
            page_tables,
            num_channels,
            num_resolutions,
        }
    }

    // todo: find a better name
    pub fn get_page_table_directory_shape(&self) -> UVec2 {
        UVec2::new(self.num_channels as u32, self.num_resolutions as u32)
    }

    pub fn get_page_table(&self, resolution: u32, channel: u32) -> &PageTableMeta {
        let subscript = UVec2::new(channel, resolution);
        let size = self.get_page_table_directory_shape();
        &self.page_tables[subscript.to_index(&size) as usize]
    }

    pub fn brick_size(&self) -> UVec3 {
        self.brick_size
    }
    pub fn extent(&self) -> UVec3 {
        self.extent
    }
    pub fn normalized_volume_size(&self) -> Vec3 {
        self.normalized_volume_size
    }
    pub fn page_tables(&self) -> &Vec<PageTableMeta> {
        &self.page_tables
    }
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
    pub fn num_resolutions(&self) -> usize {
        self.num_resolutions
    }
}
