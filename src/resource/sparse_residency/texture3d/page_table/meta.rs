use crate::volume::{BrickedMultiResolutionMultiVolumeMeta, ResolutionMeta};
use glam::{UVec2, UVec3, Vec3};
use crate::util::extent::{SubscriptToIndex};

#[repr(C)]
#[derive(Clone)]
pub struct PageTableMeta {
    /// The offset of this page table in the page directory.
    pub(crate) offset: UVec3,

    /// The extent of this page table in the page directory.
    /// The extent of the full volume represented by this page table is the component-wise product
    /// `self.extent * brick_size`, where `brick_size` is the size of one page in cache.
    pub(crate) extent: UVec3,

    ///
    pub(crate) volume_meta: ResolutionMeta,
}

impl PageTableMeta {
    pub fn new(offset: UVec3, extent: UVec3, volume_meta: ResolutionMeta) -> Self {
        Self {
            offset,
            extent,
            volume_meta,
        }
    }

    pub fn get_max_location(&self) -> UVec3 {
        self.offset + self.extent
    }
}

#[derive(Clone)]
pub struct PageDirectoryMeta {
    /// The size of a brick in the brick cache. This is constant across all resolutions of the
    /// bricked multi-resolution volume.
    pub(crate) brick_size: UVec3,

    pub(crate) scale: Vec3,

    pub(crate) extent: UVec3,

    /// The resolutions
    pub(crate) page_tables: Vec<PageTableMeta>,

    num_channels: usize,
    num_resolutions: usize,
}

impl PageDirectoryMeta {
    // todo: configure how many channels the page table can hold
    // todo: more efficient packing strategy
    pub fn new(volume_meta: &BrickedMultiResolutionMultiVolumeMeta) -> Self {
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
            let extent= volume_meta.bricks_per_dimension(level);
            for _ in 0..volume_meta.channels.len() {
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

        let extent = page_tables
            .iter()
            .fold(UVec3::ZERO, |a, b| a.max(b.get_max_location()));

        Self {
            brick_size: UVec3::from(volume_meta.brick_size),
            scale: Vec3::from(volume_meta.scale),
            extent,
            page_tables,
            num_channels: volume_meta.channels.len(),
            num_resolutions: volume_meta.resolutions.len(),
        }
    }

    pub fn get_page_table(&self, resolution: u32, channel: u32) -> &PageTableMeta {
        let subscript = UVec2::new(channel, resolution);
        let size = UVec2::new(self.num_channels as u32, self.num_resolutions as u32);
        &self.page_tables[subscript.to_index(&size) as usize]
    }
}
