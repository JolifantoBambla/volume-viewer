// todo: create mask from list & time stamp
// todo: sort list w.r.t. mask

mod lru_update;

use std::borrow::Cow;
use std::mem::size_of;
use std::sync::Arc;
use glam::UVec3;
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgsl_preprocessor::WGSLPreprocessor;

use crate::gpu_list::GpuList;
use crate::GPUContext;
use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::{MappableBuffer, Texture};
use crate::util::extent::{box_volume, extent_to_uvec, uvec_to_extent};

use lru_update::LRUUpdate;
use crate::resource::sparse_residency::texture3d::cache_management::lru::lru_update::Resources;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NumUsedEntries {
    num: u32,
}

pub struct LRUCache {
    cache: Texture,

    /// Stores one timestamp for each brick in the bricked multi-resolution hierarchy.
    usage_buffer: Texture,

    lru_buffer: Buffer,
    lru_read_buffer: MappableBuffer<u32>,
    lru_buffer_size: BufferAddress,

    num_used_entries_buffer: Buffer,
    num_used_entries_read_buffer: MappableBuffer<u32>,

    lru_update_pass: LRUUpdate,
    lru_update_bind_group: BindGroup,
}

impl LRUCache {
    pub fn new(cache_size: UVec3, entry_size: UVec3, timestamp_uniform_buffer: &Buffer, wgsl_preprocessor: &WGSLPreprocessor, ctx: &Arc<GPUContext>) -> Self {
        // todo: make configurable
        let cache = Texture::create_brick_cache(
            &ctx.device,
            uvec_to_extent(&cache_size),
        );

        let entries_per_dimension = cache_size / entry_size;
        let num_entries = box_volume(&entries_per_dimension);
        let usage_meta = vec![0u32; num_entries as usize];
        let num_unused_entries = NumUsedEntries {
            num: num_entries,
        };
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
            contents: bytemuck::cast_slice(usage_meta.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let num_used_entries_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&num_unused_entries),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let lru_read_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(usage_meta.as_slice()),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });
        let num_used_entries_read_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&num_unused_entries),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

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
            lru_buffer,
            lru_read_buffer: MappableBuffer::new(lru_read_buffer),
            lru_buffer_size,
            num_used_entries_buffer,
            num_used_entries_read_buffer: MappableBuffer::new(num_used_entries_read_buffer),
            lru_update_pass,
            lru_update_bind_group,
        }
    }

    pub fn encode_lru_update(&self, encoder: &mut CommandEncoder) {
        self.lru_update_pass.encode(
            encoder,
            &self.lru_update_bind_group,
            &self.usage_buffer.extent,
        );
        self.encode_reading(encoder);
    }

    fn encode_reading(&self, encoder: &mut CommandEncoder) {
        self.copy_to_readable(encoder);

    }

    // todo: double buffering
    fn copy_to_readable(&self, encoder: &mut CommandEncoder) {
        if self.lru_read_buffer.is_ready() && self.num_used_entries_read_buffer.is_ready() {
            encoder.copy_buffer_to_buffer(
                &self.num_used_entries_buffer,
                0,
                self.num_used_entries_read_buffer.as_buffer_ref(),
                0,
                size_of::<NumUsedEntries>() as BufferAddress,
            );
            encoder.copy_buffer_to_buffer(
                &self.lru_buffer,
                0,
                self.lru_read_buffer.as_buffer_ref(),
                0,
                self.lru_buffer_size,
            );
        }
    }
}
