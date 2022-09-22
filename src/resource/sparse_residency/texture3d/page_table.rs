use crate::resource::sparse_residency::texture3d::volume_meta::{
    MultiResolutionVolumeMeta, VolumeResolutionMeta,
};
use glam::{UVec3, UVec4, Vec3};

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PageTableEntryFlag {
    Unmapped = 0,
    Mapped = 1,
    Empty = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PageTableEntry {
    /// The 3D texture coordinate of the brick referenced by this `PageTableEntry` in the brick
    /// cache.
    /// Note: this is only valid if `flag` is `PageTableEntryFlag::Mapped`
    pub location: UVec3,

    /// A flag signalling if the brick referenced by this `PageTableEntry` is present (`PageTableEntryFlag::Mapped`),
    /// not present and possible non-empty (`PageTableEntryFlag::Unmapped`), or possibly present but
    /// does not hold any meaningful values w.r.t. the current parameters (e.g. transfer function,
    /// threshold, ...) (`PageTableEntryFlag::Empty`).
    pub flag: PageTableEntryFlag,
}

impl PageTableEntry {
    pub fn new(location: UVec3, flag: PageTableEntryFlag) -> Self {
        Self { location, flag }
    }
}

impl Default for PageTableEntry {
    fn default() -> Self {
        UVec4::default().into()
    }
}

impl From<UVec4> for PageTableEntry {
    fn from(v: UVec4) -> Self {
        Self {
            location: v.truncate(),
            flag: match v.w {
                0 => PageTableEntryFlag::Unmapped,
                1 => PageTableEntryFlag::Mapped,
                2 => PageTableEntryFlag::Empty,
                _ => {
                    log::warn!("got unknown page table entry flag value {}", v.w);
                    PageTableEntryFlag::Unmapped
                }
            },
        }
    }
}

impl From<PageTableEntry> for UVec4 {
    fn from(page_table_entry: PageTableEntry) -> Self {
        page_table_entry
            .location
            .extend(page_table_entry.flag as u32)
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct PageTableMeta {
    /// The offset of this resolution's page table in the page directory.
    offsets: Vec<UVec3>,

    pub(crate) extent: UVec3,

    ///
    pub(crate) volume_meta: VolumeResolutionMeta,
}

impl PageTableMeta {
    pub fn new(offsets: Vec<UVec3>, extent: UVec3, volume_meta: VolumeResolutionMeta) -> Self {
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
    pub fn new(volume_meta: &MultiResolutionVolumeMeta) -> Self {
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
            for _ in 0..1 { //volume_meta.channels.len() {
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

#[derive(Clone, Copy)]
pub struct PageTableAddress {
    pub(crate) location: UVec3,
    pub(crate) level: u32,
}

impl From<PageTableAddress> for [u32; 4] {
    fn from(page_table_address: PageTableAddress) -> Self {
        [
            page_table_address.location.x,
            page_table_address.location.y,
            page_table_address.location.z,
            page_table_address.level,
        ]
    }
}

impl From<[u32; 4]> for PageTableAddress {
    fn from(data: [u32; 4]) -> Self {
        Self {
            location: UVec3::new(data[0], data[1], data[2]),
            level: data[3],
        }
    }
}
