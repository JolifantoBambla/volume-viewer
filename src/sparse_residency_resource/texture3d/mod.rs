pub mod data_source;
pub mod page_table;
pub mod volume_meta;

use crate::renderer::resources::Texture;
use crate::sparse_residency_resource::texture3d::data_source::{
    Brick, SparseResidencyTexture3DSource,
};
use crate::sparse_residency_resource::texture3d::page_table::{
    PageDirectoryMeta, PageTableAddress,
};
use crate::util::extent::{extent_to_uvec, uvec_to_extent};
use std::collections::HashMap;
use wgpu::{Device, Queue};

/// Manages a 3D sparse residency texture.
/// A sparse residency texture is not necessarily present in GPU memory as a whole.
pub struct SparseResidencyTexture3D<T>
where
    T: SparseResidencyTexture3DSource,
{
    meta: PageDirectoryMeta,
    page_directory: Texture,
    brick_cache: Texture,
    brick_usage_buffer: Texture,
    request_buffer: Texture,
    local_brick_cache: HashMap<PageTableAddress, Brick>,
    source: T,
}

impl<T: SparseResidencyTexture3DSource> SparseResidencyTexture3D<T> {
    pub fn new(source: T, device: &Device, queue: &Queue) -> Self {
        let volume_meta = source.get_meta();
        let meta = PageDirectoryMeta::new(volume_meta);

        // 1 page table entry per brick
        let page_directory =
            Texture::create_page_directory(device, queue, uvec_to_extent(meta.extent));

        let brick_cache = Texture::create_brick_cache(device);

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
            page_directory,
            brick_cache,
            brick_usage_buffer,
            request_buffer,
            local_brick_cache: HashMap::new(),
            source,
        }
    }

    /// Call this after rendering has completed to read back requests & usages
    pub fn post_render(&self) {
        // request bricks
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

    pub fn request_bricks(&self) {
        //
    }
}
