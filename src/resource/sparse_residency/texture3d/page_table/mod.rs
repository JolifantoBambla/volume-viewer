pub mod meta;

use glam::{UVec3, UVec4, Vec3};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindingResource, Buffer, BufferUsages, Extent3d};

use crate::resource::Texture;
use crate::util::extent::{subscript_to_index, uvec_to_extent, IndexToSubscript};
use crate::volume::{BrickAddress, BrickedMultiResolutionMultiVolumeMeta};
use crate::GPUContext;

use crate::resource::buffer::TypedBuffer;
pub use meta::{PageDirectoryMeta, PageTableMeta};

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PageTableEntryFlag {
    Unmapped = 0,
    Mapped = 1,
    Empty = 2,
}

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
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PageDirectoryMetaGPUData {
    brick_size: UVec4,
    max_resolutions: u32,
    max_visible_channels: u32,
    padding1: u32,
    padding2: u32,
}

pub struct PageTableDirectory {
    ctx: Arc<GPUContext>,
    local_page_directory: Vec<UVec4>,
    page_directory_meta_buffer: TypedBuffer<PageDirectoryMetaGPUData>,
    page_table_meta_buffer: Buffer,
    page_directory: Texture,
    meta: PageDirectoryMeta,
    max_visible_channels: u32,
    max_resolutions: u32,

    cache_addresses_in_use: HashMap<UVec3, usize>,
}

impl PageTableDirectory {
    pub fn new(
        volume_meta: &BrickedMultiResolutionMultiVolumeMeta,
        max_visible_channels: u32,
        max_resolutions: u32,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let meta = PageDirectoryMeta::new(
            volume_meta,
            max_visible_channels as usize,
            max_resolutions as usize,
        );

        let res_meta_data: Vec<ResMeta> = meta
            .page_tables()
            .iter()
            .map(|pt| ResMeta {
                brick_size: meta.brick_size().extend(0),
                page_table_offset: pt.offset.extend(0),
                page_table_extent: pt.extent.extend(0),
                volume_size: pt.volume_meta.volume_size.extend(0),
            })
            .collect();

        for r in &res_meta_data {
            log::info!("resolution meta data: {:?}", r);
        }

        let page_directory_meta_buffer = TypedBuffer::new_single_element(
            "Page Directory Meta",
            PageDirectoryMetaGPUData {
                brick_size: meta.brick_size().extend(0),
                max_visible_channels: meta.num_channels() as u32,
                max_resolutions: meta.num_resolutions() as u32,
                ..Default::default()
            },
            BufferUsages::UNIFORM,
            &ctx.device,
        );

        let page_table_meta_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Page Table Meta"),
            contents: bytemuck::cast_slice(res_meta_data.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // 1 page table entry per brick
        let (page_directory, local_page_directory) =
            Texture::create_page_directory(&ctx.device, &ctx.queue, uvec_to_extent(&meta.extent()));

        Self {
            ctx: ctx.clone(),
            max_visible_channels,
            max_resolutions,
            page_directory_meta_buffer,
            page_table_meta_buffer,
            page_directory,
            local_page_directory,
            meta,
            cache_addresses_in_use: HashMap::new(),
        }
    }

    /// Maps a given 1D `page_index` to a 5D brick address in the page table.
    ///
    /// # Arguments
    ///
    /// * `page_index`: the 1D index to map to a brick address
    ///
    /// returns: BrickAddress
    pub fn page_index_to_brick_address(&self, page_index: u32) -> BrickAddress {
        let bytes: [u8; 4] = page_index.to_be_bytes();

        let size = self.meta.get_page_table_size();
        let subscript = size.index_to_subscript(bytes[3] as u32);

        BrickAddress::new(
            UVec3::new(bytes[0] as u32, bytes[1] as u32, bytes[2] as u32),
            subscript.x,
            subscript.y,
        )
    }

    /// Maps a given 5D `brick_address` to a 1D index into the page table.
    ///
    /// # Arguments
    ///
    /// * `brick_address`: the 5D address to map to a 1D index
    ///
    /// returns: usize
    fn brick_address_to_page_index(&self, brick_address: &BrickAddress) -> usize {
        let page_table = self
            .meta
            .get_page_table(brick_address.level, brick_address.channel);
        let offset = page_table.offset;
        let location = brick_address.index;
        subscript_to_index(&(offset + location), &self.page_directory.extent) as usize
    }

    /// Marks the brick at the given `brick_address` as empty, i.e., `PageTableEntryFlag::Empty`.
    ///
    /// # Arguments
    ///
    /// * `brick_address`: the local address of the brick to mark as empty.
    pub fn mark_as_empty(&mut self, brick_address: &BrickAddress) {
        let index = self.brick_address_to_page_index(brick_address);
        self.local_page_directory[index] = UVec3::ZERO.extend(PageTableEntryFlag::Empty as u32);
    }

    fn mark_as_mapped(&mut self, page_index: usize, cache_address: &UVec3) {
        self.local_page_directory[page_index] =
            cache_address.extend(PageTableEntryFlag::Mapped as u32);
    }

    fn mark_as_unmapped(&mut self, page_index: usize) {
        self.local_page_directory[page_index] =
            UVec3::ZERO.extend(PageTableEntryFlag::Unmapped as u32);
    }

    /// Marks the brick with `brick_address`as its local address in the page table (as opposed to
    /// its global address in the volume) as mapped, i.e., `PageTableEntryFlag::Mapped`, to the
    /// given `cache_address`.
    /// If another brick was previously mapped to the same cache address, i.e., another brick has
    /// been overridden in the cache, it is marked as unmapped, i.e.,
    /// `PageTableEntryFlag::Unmapped`, and its local address in the page table is returned.
    ///
    /// # Arguments
    ///
    /// * `brick_address`: the brick's local address in the page table
    /// * `cache_address`: the address in the cache where the brick's data is stored
    ///
    /// returns: Option<BrickAddress>
    pub fn map_brick(
        &mut self,
        brick_address: &BrickAddress,
        cache_address: &UVec3,
    ) -> Option<BrickAddress> {
        let brick_index = self.brick_address_to_page_index(brick_address);
        self.mark_as_mapped(brick_index, cache_address);
        let unmapped_brick_address =
            if let Some(&unmapped_brick_id) = self.cache_addresses_in_use.get(cache_address) {
                self.mark_as_unmapped(unmapped_brick_id);
                Some(self.page_index_to_brick_address(unmapped_brick_id as u32))
            } else {
                None
            };
        self.cache_addresses_in_use
            .insert(*cache_address, brick_index);
        unmapped_brick_address
    }

    /// Invalidates all of a channel's page tables by calling
    /// `PageTableDirectory::invalidate_page_table` on all of them.
    ///
    /// # Arguments
    ///
    /// * `channel_index`: the index of the channel for which all page tables should be invalidated
    pub fn invalidate_channel_page_tables(&mut self, channel_index: u32) {
        for resolution in 0..self.max_resolutions {
            self.invalidate_page_table(resolution, channel_index);
        }
    }

    /// Invalidates the page table with `resolution_index` and `channel_index` by marking all their
    /// pages as empty, i.e, `PageTableEntryFlag::Unmapped`.
    ///
    /// # Arguments
    ///
    /// * `resolution_index`: the resolution index of the page table to invalidate
    /// * `channel_index`: the channel index of the page table to invalidate
    pub fn invalidate_page_table(&mut self, resolution_index: u32, channel_index: u32) {
        let page_table = self.meta.get_page_table(resolution_index, channel_index);
        let offset = page_table.offset;
        let last = offset + page_table.extent;

        let begin = subscript_to_index(&(offset), &self.page_directory.extent) as usize;
        let end = subscript_to_index(&last, &self.page_directory.extent) as usize;

        for index in begin..end {
            self.mark_as_unmapped(index);
        }
    }

    /// Commits local changes to the page table directory to its GPU texture representation.
    /// Note that texture writes are delayed to the beginning of the next frame internally.
    pub fn commit_changes(&self) {
        self.page_directory
            .write(self.local_page_directory.as_slice(), &self.ctx);
    }

    pub fn volume_scale(&self) -> Vec3 {
        let size = Vec3::new(
            self.meta.page_tables()[0].volume_meta.volume_size[0] as f32,
            self.meta.page_tables()[0].volume_meta.volume_size[1] as f32,
            self.meta.page_tables()[0].volume_meta.volume_size[2] as f32,
        );
        size * (self.meta.scale() / self.meta.scale().max_element())
    }

    pub fn get_page_directory_meta_as_binding_resource(&self) -> BindingResource {
        self.page_directory_meta_buffer.buffer().as_entire_binding()
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
