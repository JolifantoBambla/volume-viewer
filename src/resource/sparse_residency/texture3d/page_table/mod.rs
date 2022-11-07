pub mod meta;

use glam::{UVec3, UVec4, Vec3};
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindingResource, Buffer, BufferUsages, Extent3d};

use crate::resource::Texture;
use crate::util::extent::{IndexToSubscript, subscript_to_index, uvec_to_extent};
use crate::volume::{Brick, BrickAddress, BrickedMultiResolutionMultiVolumeMeta, ResolutionMeta};
use crate::GPUContext;

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

// okay, so this is how this should look like:
//   a page directory holds n page tables
//   each page table belongs to a resolution and a channel
//   a page directory can hold at most m channels

// todo: rename to pagetablemeta
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResMeta {
    // note: brick size is just copied
    brick_size: UVec4,
    page_table_offset: UVec4,
    page_table_extent: UVec4,
    volume_size: UVec4,
    // todo: I also need:
    //   - volume to padded ratio
    //   - linear offset in bricks
    //   - a variable saying which channel this page table holds
}

pub struct PageTableDirectory {
    ctx: Arc<GPUContext>,
    local_page_directory: Vec<UVec4>,
    page_table_meta_buffer: Buffer,
    page_directory: Texture,
    meta: PageDirectoryMeta,
    max_visible_channels: u32,
    max_resolutions: u32,
    volume_meta: BrickedMultiResolutionMultiVolumeMeta,
}

impl PageTableDirectory {
    pub fn new(
        volume_meta: &BrickedMultiResolutionMultiVolumeMeta,
        max_visible_channels: u32,
        max_resolutions: u32,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let meta = PageDirectoryMeta::new(volume_meta);

        let res_meta_data: Vec<ResMeta> = meta
            .page_tables
            .iter()
            .map(|pt| ResMeta {
                brick_size: meta.brick_size.extend(0),
                page_table_offset: pt.offset.extend(0),
                page_table_extent: pt.extent.extend(0),
                volume_size: UVec3::from_slice(pt.volume_meta.volume_size.as_slice()).extend(0),
            })
            .collect();

        for r in &res_meta_data {
            log::info!("resolution meta data: {:?}", r);
        }

        let page_table_meta_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Page Table Meta"),
            contents: bytemuck::cast_slice(res_meta_data.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // 1 page table entry per brick
        let (page_directory, local_page_directory) =
            Texture::create_page_directory(&ctx.device, &ctx.queue, uvec_to_extent(&meta.extent));

        Self {
            ctx: ctx.clone(),
            max_visible_channels,
            max_resolutions,
            volume_meta: volume_meta.clone(),
            page_table_meta_buffer,
            page_directory,
            local_page_directory,
            meta,
        }
    }

    /// Translates a `page_index` created at time `timestamp` for the GPU-resident cache to an
    /// address in the bricked multi-resolution muli-volume hierarchy represented by this page
    /// directory.
    ///
    /// # Arguments
    ///
    /// * `page_index`:
    /// * `timestamp`: CURRENTLY UNUSED! - later this will be used to map an index to the active channel-level configuration in the time interval containing the timestamp
    ///
    /// returns: BrickAddress
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn page_index_to_address(&self, page_index: u32, timestamp: u32) -> BrickAddress {
        // todo: find out why these are in big endian - my system is little endian AND webgpu ensures little endian
        let bytes: [u8; 4] = page_index.to_be_bytes();

        // todo: as soon as timestamp is used, `size` should come from a map of timestamps to channel-resolution configurations
        let size = self.meta.get_page_table_size();
        let subscript = size.index_to_subscript(bytes[3] as u32);

        BrickAddress::new(
            [bytes[0] as u32, bytes[1] as u32, bytes[2] as u32],
            subscript.x,
            subscript.y,
        )
    }

    fn brick_address_to_page_index(&self, brick_address: &BrickAddress) -> usize {
        let page_table = self.meta.get_page_table(brick_address.level, brick_address.channel);
        let offset = page_table.offset;
        let location = UVec3::from_slice(brick_address.index.as_slice());
        let page_index =
            subscript_to_index(&(offset + location), &self.page_directory.extent) as usize;
        page_index
    }

    pub fn mark_as_empty(&mut self, brick_address: &BrickAddress) {
        let index = self.brick_address_to_page_index(brick_address);
        self.local_page_directory[index] = UVec3::ZERO.extend(PageTableEntryFlag::Empty as u32);
    }

    pub fn mark_as_mapped(&mut self, brick_address: &BrickAddress, brick_location: &UVec3) {
        let index = self.brick_address_to_page_index(brick_address);
        self.local_page_directory[index] = brick_location.extend(PageTableEntryFlag::Mapped as u32);
    }

    pub fn invalidate_channel_page_tables(&mut self, channel_index: u32) {
        for resolution in 0..self.max_resolutions {
            self.invalidate_page_table(resolution, channel_index);
        }
    }

    pub fn invalidate_page_table(&mut self, resolution: u32, channel: u32) {
        // todo:
    }

    pub fn commit_changes(&self) {
        self.page_directory
            .write(self.local_page_directory.as_slice(), &self.ctx);
    }

    pub fn volume_scale(&self) -> Vec3 {
        let size = Vec3::new(
            self.meta.page_tables[0].volume_meta.volume_size[0] as f32,
            self.meta.page_tables[0].volume_meta.volume_size[1] as f32,
            self.meta.page_tables[0].volume_meta.volume_size[2] as f32,
        );
        size * (self.meta.scale / self.meta.scale.max_element())
    }

    pub fn get_page_table_meta_as_binding_resource(&self) -> BindingResource {
        self.page_table_meta_buffer.as_entire_binding()
    }

    pub fn get_page_directory_as_binding_resource(&self) -> BindingResource {
        BindingResource::TextureView(&self.page_directory.view)
    }

    pub fn page_table_meta_buffer(&self) -> &Buffer {
        &self.page_table_meta_buffer
    }

    pub fn extent(&self) -> Extent3d {
        self.page_directory.extent
    }

    pub fn channel_capacity(&self) -> usize {
        self.max_visible_channels as usize
    }
}
