// todo: create mask from list & time stamp
// todo: sort list w.r.t. mask

mod lru_update;

use glam::UVec3;
use std::borrow::Cow;
use std::mem::size_of;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder};
use wgsl_preprocessor::WGSLPreprocessor;

use crate::gpu_list::GpuList;
use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::{MappableBuffer, Texture};
use crate::util::extent::{box_volume, extent_to_uvec, uvec_to_extent};
use crate::{GPUContext, Input};

use crate::resource::buffer::MultiBufferedMappableBuffer;
use crate::resource::sparse_residency::texture3d::cache_management::lru::lru_update::Resources;
use lru_update::LRUUpdate;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NumUsedEntries {
    num: u32,
}

pub struct LRUCache {
    cache: Texture,

    /// Stores one timestamp for each brick in the bricked multi-resolution hierarchy.
    usage_buffer: Texture,

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
        entry_size: UVec3,
        timestamp_uniform_buffer: &Buffer,
        num_multi_buffering: u32,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        // todo: make configurable
        let cache = Texture::create_brick_cache(&ctx.device, uvec_to_extent(&cache_size));

        let entries_per_dimension = cache_size / entry_size;
        let num_entries = box_volume(&entries_per_dimension);
        let num_unused_entries = NumUsedEntries { num: num_entries };

        let lru_local = vec![0u32; num_entries as usize];
        let lru_buffer_size = (size_of::<u32>() * num_entries as usize) as BufferAddress;

        // 1:1 mapping, 1 timestamp per brick in cache
        let usage_buffer = Texture::create_u32_storage_3d(
            "Usage Buffer".to_string(),
            &ctx.device,
            &ctx.queue,
            uvec_to_extent(&entries_per_dimension),
        );

        let lru_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(lru_local.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let num_used_entries_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&num_unused_entries),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let lru_read_buffer = MultiBufferedMappableBuffer::new(
            num_multi_buffering,
            &lru_local,
            &ctx.device,
        );
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
            cache,
            usage_buffer,
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

    pub fn encode_lru_update(&self, encoder: &mut CommandEncoder, input: &Input) {
        self.lru_update_pass.encode(
            encoder,
            &self.lru_update_bind_group,
            &self.usage_buffer.extent,
        );
        // we can't copy to and read from the same buffer in one frame
        // so we'll copy to the current frame's buffer
        // and read from the last frame's buffer (if it is mapped, i.e. maybe_read_all)
        self.copy_to_readable(encoder, input.frame.number);
    }

    fn copy_to_readable(&self, encoder: &mut CommandEncoder, buffer_index: u32) {
        if self.lru_read_buffer.is_ready(buffer_index) && self.num_used_entries_read_buffer.is_ready(buffer_index) {
            encoder.copy_buffer_to_buffer(
                &self.num_used_entries_buffer,
                0,
                self.num_used_entries_read_buffer.as_buffer_ref(buffer_index),
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

    /// Maps the current frame's LRU read buffer for reading and tries to read the last frame's to
    /// to update its CPU local list of free cache entries.
    pub fn update_local_lru(&mut self, input: &Input) {
        self.lru_read_buffer.map_async(input.frame.number, wgpu::MapMode::Read, ..);
        self.num_used_entries_read_buffer.map_async(input.frame.number, wgpu::MapMode::Read, ..);

        let last_lru_index = self.lru_read_buffer.to_previous_index(input.frame.number);
        let last_num_entries_index = self.num_used_entries_read_buffer.to_previous_index(input.frame.number);
        if self.lru_read_buffer.is_mapped(last_lru_index) && self.num_used_entries_read_buffer.is_mapped(last_num_entries_index) {
            let lru = self.lru_read_buffer.maybe_read_all(last_lru_index);
            let num_used_entries = self.num_used_entries_read_buffer.maybe_read_all(last_lru_index);

            if lru.is_empty() || num_used_entries.is_empty() {
                log::error!("Could not read LRU at frame {}", input.frame.number);
            } else {
                self.lru_local = lru;
                self.num_used_entries_local = num_used_entries[0].num;
            }
        } else {
            log::warn!("Could not update LRU at frame {}", input.frame.number);
        }
    }

    pub(crate) fn get_cache_as_binding_resource(&self) -> BindingResource {
        BindingResource::TextureView(&self.cache.view)
    }

    pub(crate) fn get_usage_buffer_as_binding_resource(&self) -> BindingResource {
        BindingResource::TextureView(&self.usage_buffer.view)
    }

    /// Writes data to the cache and returns the 3D cache address of the slot the data was written to
    pub fn add_cache_entry(&self, data: &Vec<u8>) -> Result<UVec3, CacheFullError> {
        // todo: implement
        Ok(UVec3::ZERO)
    }
}
