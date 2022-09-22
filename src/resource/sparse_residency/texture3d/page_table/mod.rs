pub mod meta;

use glam::{UVec3, UVec4, Vec3};
use wgpu::Buffer;

use crate::resource::Texture;
use crate::volume::{BrickedMultiResolutionMultiVolumeMeta, ResolutionMeta};
pub use meta::{PageDirectoryMeta, PageTableMeta};

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

pub struct PageTableDirectory {
    local_page_directory: Vec<UVec4>,
    page_table_meta_buffer: Buffer,
    page_directory: Texture,
    meta: PageDirectoryMeta,
}

impl PageTableDirectory {
    //pub fn new(volume_meta: &MultiResolutionMultiVolumeMeta) -> Self {
    // Self {}
    //}
}
