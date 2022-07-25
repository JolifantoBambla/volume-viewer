use crate::renderer::resources::Texture;
use glam::{UVec3, UVec4, Vec3};
use wgpu::{Device, Extent3d, Queue};

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
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

impl Into<UVec4> for PageTableEntry {
    fn into(self) -> UVec4 {
        self.location.extend(self.flag as u32)
    }
}

// todo: move out of here
#[derive(Clone, Copy)]
pub struct VolumeResolutionMeta {
    /// The size of the volume in voxels.
    /// It is not necessarily a multiple of `brick_size`.
    /// See `padded_volume_size`.
    volume_size: UVec3,

    /// The size of the volume in voxels padded s.t. it is a multiple of `PageTableMeta::brick_size`.
    padded_volume_size: UVec3,

    /// The spatial extent of the volume.
    scale: Vec3,
}

// todo: move out of here
#[derive(Clone)]
pub struct MultiResolutionVolumeMeta {
    /// The size of a brick in the brick cache. This is constant across all resolutions of the
    /// bricked multi-resolution volume.
    brick_size: UVec3,

    /// The resolutions
    resolutions: Vec<VolumeResolutionMeta>,
}

impl MultiResolutionVolumeMeta {
    pub fn bricks_per_dimension(&self, level: usize) -> UVec3 {
        self.resolutions[level].padded_volume_size / self.brick_size
    }

    pub fn number_of_bricks(&self, level: usize) -> u32 {
        self.bricks_per_dimension(level)
            .to_array()
            .iter()
            .fold(1, |a, &b| a * b)
    }
}

#[derive(Clone)]
pub struct PageTableMeta {
    /// The offset of this resolution's page table in the page directory.
    offset: UVec3,

    extent: UVec3,

    ///
    volume_meta: VolumeResolutionMeta,
}

#[derive(Clone)]
pub struct PageDirectoryMeta {
    /// The size of a brick in the brick cache. This is constant across all resolutions of the
    /// bricked multi-resolution volume.
    brick_size: UVec3,

    extent: UVec3,

    /// The resolutions
    resolutions: Vec<PageTableMeta>,
}

// todo: address translation
impl PageDirectoryMeta {
    pub fn new(volume_meta: &MultiResolutionVolumeMeta) -> Self {
        let mut resolutions: Vec<PageTableMeta> = Vec::new();
        for (level, volume_resolution) in volume_meta.resolutions.iter().enumerate() {
            let offset = if level > 0 {
                let last_offset = resolutions[level - 1].offset;
                let last_extent = resolutions[level - 1].extent;

                last_offset
                    + if last_extent.x == last_extent.min_element() {
                        UVec3::new(last_extent.x, 0, 0)
                    } else if last_extent.y == last_extent.min_element() {
                        UVec3::new(0, last_extent.y, 0)
                    } else {
                        UVec3::new(0, 0, last_extent.z)
                    }
            } else {
                UVec3::ZERO
            };
            let extent = volume_meta.bricks_per_dimension(level);
            resolutions.push(PageTableMeta {
                offset,
                extent,
                volume_meta: volume_resolution.clone(),
            });
        }

        let extent = resolutions
            .iter()
            .fold(UVec3::ZERO, |a, b| a.max(b.offset + b.extent));

        Self {
            brick_size: volume_meta.brick_size,
            extent,
            resolutions,
        }
    }
}

pub struct PageTableManager {
    meta: PageDirectoryMeta,
    page_directory: Texture,
}

impl PageTableManager {
    pub fn new(
        multi_resolution_volume_meta: &MultiResolutionVolumeMeta,
        device: &Device,
        queue: &Queue,
    ) -> Self {
        let meta = PageDirectoryMeta::new(multi_resolution_volume_meta);
        let page_directory = Texture::create_page_directory(
            device,
            queue,
            Extent3d {
                width: meta.extent.x,
                height: meta.extent.y,
                depth_or_array_layers: meta.extent.z,
            },
        );
        Self {
            meta,
            page_directory,
        }
    }
}
