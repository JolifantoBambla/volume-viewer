mod lru_update;

use glam::UVec3;
use std::sync::Arc;
use wgpu::{BindingResource, Buffer, CommandEncoder, Extent3d};
use wgsl_preprocessor::WGSLPreprocessor;

use crate::resource::{buffer::ReadableStorageBuffer, Texture};
use crate::util::extent::{
    box_volume, extent_to_uvec, index_to_subscript, uvec_to_extent, uvec_to_origin,
};
use crate::util::multi_buffer::MultiBuffered;
use wgpu_framework::input::Input;

use lru_update::{LRUUpdate, LRUUpdateResources};
use wgpu_framework::context::Gpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NumUsedEntries {
    num: u32,
}

pub struct CacheFullError {}

pub struct LRUCacheSettings {
    pub cache_size: Extent3d,
    pub cache_entry_size: UVec3,
    pub num_multi_buffering: u32,
    pub time_to_live: u32,
}

#[derive(Debug)]
struct LRUCacheGpuOps {
    lru: ReadableStorageBuffer<u32>,
    num_used_entries: ReadableStorageBuffer<NumUsedEntries>,
    lru_update_pass: LRUUpdate,
}

impl LRUCacheGpuOps {
    pub fn encode(&self, command_encoder: &mut CommandEncoder, frame_number: u32) {
        self.lru_update_pass.encode(command_encoder);

        if self.lru.copy_to_readable(command_encoder).is_err() {
            log::debug!(
                "Frame {}: could not copy to readable ({})",
                frame_number,
                self.lru
            );
        }
        if self
            .num_used_entries
            .copy_to_readable(command_encoder)
            .is_err()
        {
            log::debug!(
                "Frame {}: could not copy to readable ({})",
                frame_number,
                self.num_used_entries
            );
        }
    }

    pub fn map_read_buffers(&self, frame_number: u32) {
        if self.lru.map_all_async().is_ok() {
            if self.num_used_entries.map_all_async().is_err() {
                self.lru.unmap();
                log::debug!(
                    "Frame {}: could not map ({})",
                    frame_number,
                    self.num_used_entries
                );
            }
        } else {
            log::debug!("Frame {}: could not map ({})", frame_number, self.lru);
        }
    }

    pub fn read_buffers(&self, frame_number: u32) -> Result<(Vec<u32>, NumUsedEntries), ()> {
        if let Ok(lru) = self.lru.read_all() {
            if let Ok(num_used_entries) = self.num_used_entries.read_all() {
                Ok((lru, num_used_entries[0]))
            } else {
                log::debug!(
                    "Frame {}: could not read ({})",
                    frame_number,
                    self.num_used_entries
                );
                Err(())
            }
        } else {
            log::debug!("Frame {}: could not read ({})", frame_number, self.lru);
            Err(())
        }
    }
}

#[derive(Debug)]
pub struct LRUCache {
    ctx: Arc<Gpu>,

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

    num_used_entries_local: u32,

    lru_stuff: MultiBuffered<LRUCacheGpuOps>,
}

impl LRUCache {
    pub fn new(
        settings: LRUCacheSettings,
        timestamp_uniform_buffer: &Buffer,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<Gpu>,
    ) -> Self {
        let cache = Texture::create_brick_cache(ctx.device(), settings.cache_size);

        let usage_buffer_size = extent_to_uvec(&settings.cache_size) / settings.cache_entry_size;
        let num_entries = box_volume(&usage_buffer_size);
        let num_unused_entries = NumUsedEntries { num: num_entries };

        let next_empty_index = num_entries;
        let lru_local = Vec::from_iter((0..num_entries).rev());
        let lru_last_update = vec![u32::MAX; num_entries as usize];

        // 1:1 mapping, 1 timestamp per brick in cache
        let usage_buffer = Texture::create_u32_storage_3d(
            "Usage Buffer".to_string(),
            ctx.device(),
            ctx.queue(),
            uvec_to_extent(&usage_buffer_size),
        );

        let lru_stuff = MultiBuffered::new(
            || {
                let lru = ReadableStorageBuffer::from_data("lru", &lru_local, ctx.device());
                let num_used_entries = ReadableStorageBuffer::from_data(
                    "num used entries",
                    &vec![num_unused_entries],
                    ctx.device(),
                );

                let lru_update_pass = LRUUpdate::new(
                    &LRUUpdateResources::new(
                        lru.storage_buffer().clone(),
                        num_used_entries.storage_buffer().clone(),
                        &usage_buffer,
                        timestamp_uniform_buffer,
                    ),
                    wgsl_preprocessor,
                    ctx,
                );

                LRUCacheGpuOps {
                    lru,
                    num_used_entries,
                    lru_update_pass,
                }
            },
            settings.num_multi_buffering as usize,
        );

        Self {
            ctx: ctx.clone(),
            cache,
            cache_entry_size: settings.cache_entry_size,
            usage_buffer,
            lru_last_writes: lru_last_update,
            time_to_live: settings.time_to_live,
            next_empty_index,
            lru_local,
            num_used_entries_local: 0,
            lru_stuff,
        }
    }

    pub fn encode_lru_update(&self, encoder: &mut CommandEncoder, timestamp: u32) {
        self.lru_stuff
            .get(timestamp as usize)
            .encode(encoder, timestamp);
    }

    /// tries to read the last frame's to
    /// to update its CPU local list of free cache entries.
    pub fn update_local_lru(&mut self, timestamp: u32) {
        self.lru_stuff
            .get(timestamp as usize)
            .map_read_buffers(timestamp);

        if let Ok((lru, num_used_entries)) = self
            .lru_stuff
            .get_previous(timestamp as usize)
            .read_buffers(timestamp)
        {
            self.lru_local = lru;
            self.num_used_entries_local = num_used_entries.num;
            self.next_empty_index = self.lru_local.len() as u32;
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
        for i in ((self.num_used_entries_local as usize)..(self.next_empty_index as usize)).rev() {
            let cache_entry_index = self.lru_local[i];
            let last_written = self.lru_last_writes[cache_entry_index as usize];
            if last_written > input.frame().number()
                || (input.frame().number() - last_written) > self.time_to_live
            {
                let extent = uvec_to_extent(
                    &(index_to_subscript(
                        (data.len() as u32) - 1,
                        &uvec_to_extent(&self.cache_entry_size),
                    ) + UVec3::ONE),
                );

                // todo: batch updates (technically this is batched by wgpu
                let cache_entry_location = self.cache_entry_index_to_location(cache_entry_index);
                self.cache.write_subregion(
                    data.as_slice(),
                    uvec_to_origin(&cache_entry_location),
                    extent,
                    &self.ctx,
                );
                self.lru_last_writes[cache_entry_index as usize] = input.frame().number();
                self.next_empty_index = i as u32; // if i > 0 { i as u32 - 1 } else { 0 };
                return Ok(cache_entry_location);
            }
        }
        Err(CacheFullError {})
    }

    #[allow(unused)]
    pub fn cache_entry_size(&self) -> UVec3 {
        self.cache_entry_size
    }
}
