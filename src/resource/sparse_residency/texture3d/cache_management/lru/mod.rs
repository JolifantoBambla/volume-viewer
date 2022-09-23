// todo: create mask from list & time stamp
// todo: sort list w.r.t. mask

mod lru_update;

use glam::UVec3;
use std::collections::HashSet;
use std::mem::size_of;
use std::sync::Arc;
use web_sys::console::time;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindingResource, Buffer, BufferAddress, BufferUsages, CommandEncoder, Extent3d,
};
use wgsl_preprocessor::WGSLPreprocessor;

use crate::renderer::pass::GPUPass;
use crate::resource::Texture;
use crate::util::extent::{box_volume, index_to_subscript, uvec_to_extent, uvec_to_origin};
use crate::{GPUContext, Input};

use crate::resource::buffer::MultiBufferedMappableBuffer;
use crate::resource::sparse_residency::texture3d::cache_management::lru::lru_update::Resources;
use lru_update::LRUUpdate;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NumUsedEntries {
    num: u32,
}

pub struct CacheFullError {}

pub struct LRUCache {
    ctx: Arc<GPUContext>,

    cache: Texture,
    cache_entry_size: UVec3,

    /// Stores one timestamp for each brick in the bricked multi-resolution hierarchy.
    usage_buffer: Texture,

    /// Stores the last timestamp the cache entry with the corresponding index has been written.
    /// If a cache entry has never been written to `u32::MAX` is stored as a special value.
    lru_last_writes: Vec<u32>,

    // Each cache entry can't be overridden for `time_to_live` frames.
    time_to_live: u32,

    next_empty_index: u32,

    lru_local: Vec<u32>,
    lru_buffer: Buffer,
    lru_read_buffer: MultiBufferedMappableBuffer<u32>,
    lru_buffer_size: BufferAddress,

    num_used_entries_local: u32,
    num_used_entries_buffer: Buffer,
    num_used_entries_read_buffer: MultiBufferedMappableBuffer<NumUsedEntries>,

    lru_update_pass: LRUUpdate,
    lru_update_bind_group: BindGroup,
}

impl LRUCache {
    pub fn new(
        cache_size: UVec3,
        cache_entry_size: UVec3,
        timestamp_uniform_buffer: &Buffer,
        num_multi_buffering: u32,
        time_to_live: u32,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        // todo: make configurable
        let cache = Texture::create_brick_cache(&ctx.device, uvec_to_extent(&cache_size));

        let usage_buffer_size = cache_size / cache_entry_size;
        let num_entries = box_volume(&usage_buffer_size);
        let num_unused_entries = NumUsedEntries { num: num_entries };

        let next_empty_index = num_entries;
        let lru_local = Vec::from_iter((0..num_entries).rev());
        let lru_last_update = vec![u32::MAX; num_entries as usize];
        let lru_buffer_size = (size_of::<u32>() * num_entries as usize) as BufferAddress;

        // 1:1 mapping, 1 timestamp per brick in cache
        let usage_buffer = Texture::create_u32_storage_3d(
            "Usage Buffer".to_string(),
            &ctx.device,
            &ctx.queue,
            uvec_to_extent(&usage_buffer_size),
        );

        let lru_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(lru_local.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let num_used_entries_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&num_unused_entries),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let lru_read_buffer =
            MultiBufferedMappableBuffer::new(num_multi_buffering, &lru_local, &ctx.device);
        let num_used_entries_read_buffer = MultiBufferedMappableBuffer::new(
            num_multi_buffering,
            &vec![num_unused_entries],
            &ctx.device,
        );

        let lru_update_pass = LRUUpdate::new(num_entries, wgsl_preprocessor, ctx);
        let lru_update_bind_group = lru_update_pass.create_bind_group(Resources {
            usage_buffer: &usage_buffer,
            timestamp: timestamp_uniform_buffer,
            lru_cache: &lru_buffer,
            num_used_entries: &num_used_entries_buffer,
        });

        Self {
            ctx: ctx.clone(),
            cache,
            cache_entry_size,
            usage_buffer,
            lru_last_writes: lru_last_update,
            time_to_live,
            next_empty_index,
            lru_local,
            lru_buffer,
            lru_read_buffer,
            lru_buffer_size,
            num_used_entries_local: 0,
            num_used_entries_buffer,
            num_used_entries_read_buffer,
            lru_update_pass,
            lru_update_bind_group,
        }
    }

    pub fn encode_lru_update(&self, encoder: &mut CommandEncoder, timestamp: u32) {
        let num_used_entries = NumUsedEntries { num: 0 };
        self.ctx.queue.write_buffer(
            &self.num_used_entries_buffer,
            0 as BufferAddress,
            bytemuck::bytes_of(&num_used_entries),
        );

        self.lru_update_pass.encode(
            encoder,
            &self.lru_update_bind_group,
            &self.usage_buffer.extent,
        );
        self.copy_to_readable(encoder, timestamp + 1);
    }

    fn copy_to_readable(&self, encoder: &mut CommandEncoder, buffer_index: u32) {
        if self.lru_read_buffer.is_ready(buffer_index)
            && self.num_used_entries_read_buffer.is_ready(buffer_index)
        {
            encoder.copy_buffer_to_buffer(
                &self.num_used_entries_buffer,
                0,
                self.num_used_entries_read_buffer
                    .as_buffer_ref(buffer_index),
                0,
                size_of::<NumUsedEntries>() as BufferAddress,
            );
            encoder.copy_buffer_to_buffer(
                &self.lru_buffer,
                0,
                self.lru_read_buffer.as_buffer_ref(buffer_index),
                0,
                self.lru_buffer_size,
            );
        }
    }

    /// tries to read the last frame's to
    /// to update its CPU local list of free cache entries.
    pub fn update_local_lru(&mut self, timestamp: u32) {
        let next_frame = timestamp + 1;

        //  Maps the current frame's LRU read buffer for reading
        self.lru_read_buffer
            .map_async(next_frame, wgpu::MapMode::Read, ..);
        self.num_used_entries_read_buffer
            .map_async(next_frame, wgpu::MapMode::Read, ..);

        if self.lru_read_buffer.is_mapped(timestamp)
            && self.num_used_entries_read_buffer.is_mapped(timestamp)
        {
            let lru = self.lru_read_buffer.maybe_read_all(timestamp);
            let num_used_entries = self.num_used_entries_read_buffer.maybe_read_all(timestamp);

            if lru.is_empty() || num_used_entries.is_empty() {
                log::error!("Could not read LRU at frame {}", timestamp);
            } else {
                let h: HashSet<u32> = HashSet::from_iter(lru.iter().cloned());
                assert_eq!(
                    h.len(),
                    lru.len(),
                    "lru {:?}, num_used: {}",
                    lru,
                    num_used_entries[0].num
                );
                //log::info!("updated lru {:?}", lru);
                self.lru_local = lru;
                self.num_used_entries_local = num_used_entries[0].num;
                self.next_empty_index = self.lru_local.len() as u32;
            }
        } else {
            log::warn!("Could not update LRU at frame {}", timestamp);
        }
    }

    pub(crate) fn get_cache_as_binding_resource(&self) -> BindingResource {
        BindingResource::TextureView(&self.cache.view)
    }

    pub(crate) fn get_usage_buffer_as_binding_resource(&self) -> BindingResource {
        BindingResource::TextureView(&self.usage_buffer.view)
    }

    pub fn num_writable_bricks(&self) -> usize {
        // todo: filter with ttl? worst case: would loop over 32768 entries in a 1024³ cache with 32^3 brick size
        self.lru_local.len() - self.num_used_entries_local as usize
    }

    fn cache_entry_index_to_location(&self, index: u32) -> UVec3 {
        index_to_subscript(index, &self.usage_buffer.extent) * self.cache_entry_size
    }

    /// Writes data to the cache and returns the 3D cache address of the slot the data was written to
    pub fn add_cache_entry(
        &mut self,
        data: &Vec<u8>,
        input: &Input,
    ) -> Result<UVec3, CacheFullError> {
        let mut last_i = (self.next_empty_index + 1) as usize;
        for i in ((self.num_used_entries_local as usize)..(self.next_empty_index as usize)).rev() {
            let cache_entry_index = self.lru_local[i];
            let last_written = self.lru_last_writes[cache_entry_index as usize];
            if last_written > input.frame.number
                || (input.frame.number - last_written) > self.time_to_live
            {
                log::info!("writing idx {}", cache_entry_index);
                let extent = uvec_to_extent(
                    &(index_to_subscript(
                        (data.len() as u32) - 1,
                        &uvec_to_extent(&self.cache_entry_size),
                    ) + UVec3::ONE),
                );

                let cache_entry_location = self.cache_entry_index_to_location(cache_entry_index);
                self.cache.write_subregion(
                    data.as_slice(),
                    uvec_to_origin(&cache_entry_location),
                    extent,
                    &self.ctx,
                );
                self.lru_last_writes[cache_entry_index as usize] = input.frame.number;
                self.next_empty_index = i as u32; // if i > 0 { i as u32 - 1 } else { 0 };
                return Ok(cache_entry_location);
            }
        }
        Err(CacheFullError {})
    }

    pub fn cache_entry_size(&self) -> UVec3 {
        self.cache_entry_size
    }
}
