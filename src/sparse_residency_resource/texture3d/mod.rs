pub mod data_source;
pub mod page_table;
pub mod volume_meta;

use std::cmp::min;
use crate::renderer::resources::Texture;
use crate::sparse_residency_resource::texture3d::data_source::{
    Brick, SparseResidencyTexture3DSource,
};
use crate::sparse_residency_resource::texture3d::page_table::{
    PageDirectoryMeta, PageTableAddress,
};
use crate::util::extent::{extent_to_uvec, uvec_to_extent};
use std::collections::HashMap;
use glam::{UVec3, UVec4, Vec3};
use wgpu::{BindGroupEntry, BindingResource, Buffer, BufferDescriptor, BufferUsages, Device, Extent3d, Maintain, MaintainBase, Queue};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use crate::renderer::pass::AsBindGroupEntries;

/// Manages a 3D sparse residency texture.
/// A sparse residency texture is not necessarily present in GPU memory as a whole.
pub struct SparseResidencyTexture3D {
    meta: PageDirectoryMeta,
    local_brick_cache: HashMap<PageTableAddress, Brick>,
    source: Box<dyn SparseResidencyTexture3DSource>,
    brick_transfer_limit: usize,
    brick_request_limit: usize,

    // GPU resources
    page_table_meta_buffer: Buffer,
    page_directory: Texture,
    brick_cache: Texture,
    brick_usage_buffer: Texture,
    request_buffer: Texture,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResMeta {
    // note: brick size is just copied
    brick_size: UVec4,
    page_table_offset: UVec4,
    page_table_extent: UVec4,
    volume_size: UVec4,
}

impl SparseResidencyTexture3D {
    pub fn new(source: Box<dyn SparseResidencyTexture3DSource>, device: &Device, queue: &Queue) -> Self {
        let volume_meta = source.get_meta();
        let meta = PageDirectoryMeta::new(volume_meta);

        let res_meta_data: Vec<ResMeta> = meta.resolutions.iter().map(|r| ResMeta {
            brick_size: meta.brick_size.extend(0),
            page_table_offset: r.offset.extend(0),
            page_table_extent: r.extent.extend(0),
            volume_size: UVec3::from_slice(r.volume_meta.volume_size.as_slice()).extend(0)
        }).collect();

        log::info!("create pt meta buffer");
        let page_table_meta_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Page Table Meta"),
            contents: &bytemuck::cast_slice(res_meta_data.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        log::info!("created pt meta buffer");

        // 1 page table entry per brick
        let page_directory =
            Texture::create_page_directory(device, queue, uvec_to_extent(meta.extent));

        // todo: make configurable
        let brick_cache = Texture::create_brick_cache(device, Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1024,
        });

        let brick_cache_size = extent_to_uvec(brick_cache.extent);
        let bricks_per_dimension = brick_cache_size / meta.brick_size;

        // 1:1 mapping, 1 timestamp per brick in cache
        let brick_usage_buffer = Texture::create_u32_storage_3d(
            "Usage Buffer".to_string(),
            device,
            queue,
            uvec_to_extent(bricks_per_dimension),
        );

        // 1:1 mapping, 1 timestamp per brick in multi-res volume
        let request_buffer = Texture::create_u32_storage_3d(
            "Request Buffer".to_string(),
            device,
            queue,
            uvec_to_extent(meta.extent),
        );

        Self {
            meta,
            local_brick_cache: HashMap::new(),
            source,
            brick_transfer_limit: 32,
            brick_request_limit: 32,
            page_table_meta_buffer,
            page_directory,
            brick_cache,
            brick_usage_buffer,
            request_buffer,
        }
    }

    pub fn extent_f32(&self) -> Vec3 {
        Vec3::new(
            self.meta.extent.x as f32,
            self.meta.extent.y as f32,
            self.meta.extent.z as f32,
        )
    }

    /// Call this after rendering has completed to read back requests & usages
    pub fn update_cache(&mut self, device: &Device) {
        // find unused
        // find requested

        // todo: map buffers
        // buffers are mapped async, but we can't execute futures in synchronous code
        // so we need to synchronize the CPU and GPU here!
        device.poll(Maintain::Wait);
        // todo: read and unmap buffers

        let free_slots: Vec<usize> = Vec::new();

        let new_bricks = self.source.poll_bricks(min(self.brick_transfer_limit, free_slots.len()));

        // filter requested not in new_bricks
        // request
        // update cache
    }

    fn find_unused_bricks(&self) {
        // go through usage buffer and find where timestamp = now
        // for all of those which haven't been used in this
    }

    pub fn add_new_brick(&self) {
        // find location in brick cache where to add
        // write brick to brick_cache
        // write page entry to page_directory
    }

    pub fn request_bricks(&mut self) {
        self.source.request_bricks(vec![PageTableAddress::from([0, 0, 0, 2])])
    }
}

impl AsBindGroupEntries for SparseResidencyTexture3D {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.page_table_meta_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&self.page_directory.view)
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&self.brick_cache.view)
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&self.brick_usage_buffer.view)
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&self.request_buffer.view)
            },
        ]
    }
}
