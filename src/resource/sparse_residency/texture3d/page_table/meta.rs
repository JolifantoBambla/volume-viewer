use crate::volume::{BrickedMultiResolutionMultiVolumeMeta, ResolutionMeta};
use glam::{UVec3, Vec3};

#[repr(C)]
#[derive(Clone)]
pub struct PageTableMeta {
    /// The offset of this resolution's page table in the page directory.
    offsets: Vec<UVec3>,

    pub(crate) extent: UVec3,

    ///
    pub(crate) volume_meta: ResolutionMeta,
}

impl PageTableMeta {
    pub fn new(offsets: Vec<UVec3>, extent: UVec3, volume_meta: ResolutionMeta) -> Self {
        Self {
            offsets,
            extent,
            volume_meta,
        }
    }

    pub fn get_channel_offset(&self, channel: u32) -> UVec3 {
        self.offsets[channel as usize]
    }

    pub fn get_max_location(&self) -> UVec3 {
        self.offsets
            .iter()
            .fold(UVec3::ZERO, |a, &b| a.max(b + self.extent))
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
    pub(crate) resolutions: Vec<PageTableMeta>,
}

// todo: address translation
impl PageDirectoryMeta {
    pub fn new(volume_meta: &BrickedMultiResolutionMultiVolumeMeta) -> Self {
        let mut resolutions: Vec<PageTableMeta> = Vec::new();
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
            let mut offsets = Vec::new();
            // todo: configure how many channels the page table can hold
            // todo: better packing
            for _ in 0..1 {
                //volume_meta.channels.len() {
                let offset = last_offset + last_extent * packing_axis;
                offsets.push(offset);
                last_offset = offset;
                last_extent = UVec3::from_slice(&volume_resolution.volume_size);
            }
            let extent = volume_meta.bricks_per_dimension(level);
            resolutions.push(PageTableMeta::new(
                offsets,
                extent,
                volume_resolution.clone(),
            ));
        }

        let extent = resolutions
            .iter()
            .fold(UVec3::ZERO, |a, b| a.max(b.get_max_location()));

        Self {
            brick_size: UVec3::from(volume_meta.brick_size),
            scale: Vec3::from(volume_meta.scale),
            extent,
            resolutions,
        }
    }
}
