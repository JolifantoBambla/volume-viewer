mod cache_management;
mod page_table;

use glam::{UVec3, UVec4, Vec3};
use std::cmp::min;
use std::collections::HashSet;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferUsages,
    CommandEncoder, Extent3d,
};
use wgsl_preprocessor::WGSLPreprocessor;

use crate::input::Input;
use crate::renderer::context::GPUContext;
use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::Texture;
use crate::util::extent::{
    extent_to_uvec, index_to_subscript, subscript_to_index, uvec_to_extent, uvec_to_origin,
};
use crate::volume::{Brick, BrickAddress, VolumeDataSource};

use crate::resource::sparse_residency::texture3d::page_table::PageTableDirectory;
use cache_management::{
    lru::LRUCache,
    process_requests::{ProcessRequests, Resources},
    Timestamp,
};
use page_table::{PageDirectoryMeta, PageTableEntryFlag};

/// Manages a 3D sparse residency texture.
/// A sparse residency texture is not necessarily present in GPU memory as a whole.
pub struct SparseResidencyTexture3D {
    ctx: Arc<GPUContext>,

    source: Box<dyn VolumeDataSource>,
    brick_transfer_limit: usize,
    brick_request_limit: usize,

    timestamp_uniform_buffer: Buffer,

    page_table_directory: PageTableDirectory,

    lru_cache: LRUCache,

    // todo: refactor into request handler or smth.
    process_requests_pass: ProcessRequests,
    process_requests_bind_group: BindGroup,
    request_buffer: Texture,
    requested_bricks: HashSet<u32>,
    // todo: needs to be updated when cache entries are overridden
    cached_bricks: HashSet<u32>,
}

impl SparseResidencyTexture3D {
    pub fn new(
        source: Box<dyn VolumeDataSource>,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let timestamp = Timestamp::default();
        let timestamp_uniform_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&timestamp),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let volume_meta = source.get_meta();
        let brick_size = UVec3::from_slice(&volume_meta.brick_size);

        // todo: make configurable
        let max_visible_channels = 1;

        let page_table_directory = PageTableDirectory::new(volume_meta, max_visible_channels, ctx);

        // todo: make configurable
        let cache_size = Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1024,
        };
        let num_multi_buffering = 3;
        let time_to_live = u32::MAX;

        let lru_cache = LRUCache::new(
            extent_to_uvec(&cache_size),
            brick_size,
            &timestamp_uniform_buffer,
            num_multi_buffering,
            time_to_live,
            wgsl_preprocessor,
            ctx,
        );

        // 1:1 mapping, 1 timestamp per brick in multi-res volume
        let request_buffer = Texture::create_u32_storage_3d(
            "Request Buffer".to_string(),
            &ctx.device,
            &ctx.queue,
            page_table_directory.extent(),
        );

        let brick_request_limit = 32;

        let process_requests_pass =
            ProcessRequests::new(brick_request_limit as u32, wgsl_preprocessor, ctx);
        let process_requests_bind_group = process_requests_pass.create_bind_group(Resources {
            page_table_meta: &page_table_directory.page_table_meta_buffer(),
            request_buffer: &request_buffer,
            timestamp: &timestamp_uniform_buffer,
        });

        Self {
            ctx: ctx.clone(),
            source,
            brick_transfer_limit: 32,
            brick_request_limit,
            page_table_directory,
            request_buffer,
            lru_cache,
            timestamp_uniform_buffer,
            process_requests_pass,
            process_requests_bind_group,
            requested_bricks: HashSet::new(),
            cached_bricks: HashSet::new(),
        }
    }

    pub fn volume_scale(&self) -> Vec3 {
        self.page_table_directory.volume_scale()
    }

    pub fn encode_cache_management(&self, command_encoder: &mut CommandEncoder, timestamp: u32) {
        self.ctx.queue.write_buffer(
            &self.timestamp_uniform_buffer,
            0 as BufferAddress,
            bytemuck::bytes_of(&Timestamp::new(timestamp)),
        );

        self.lru_cache.encode_lru_update(command_encoder, timestamp);

        // find requested
        self.process_requests_pass.encode(
            command_encoder,
            &self.process_requests_bind_group,
            &self.request_buffer.extent,
        );
        self.process_requests_pass
            .encode_copy_result_to_readable(command_encoder);
    }

    fn process_requests(&mut self) {
        // read back requests from the GPU
        self.process_requests_pass.map_for_reading();
        let requested_ids = self.process_requests_pass.read();

        // request bricks from data source
        if !requested_ids.is_empty() {
            let mut brick_addresses =
                Vec::with_capacity(min(requested_ids.len(), self.brick_request_limit));
            for id in requested_ids {
                if !self.cached_bricks.contains(&id) && self.requested_bricks.insert(id) {
                    brick_addresses.push(BrickAddress::from(id));
                }
                if brick_addresses.len() >= self.brick_request_limit {
                    break;
                }
            }
            self.source.request_bricks(brick_addresses);
        }
    }

    fn process_new_bricks(&mut self, input: &Input) {
        // update CPU local LRU cache
        self.lru_cache.update_local_lru(input.frame.number);

        let bricks = self.source.poll_bricks(min(
            self.brick_transfer_limit,
            self.lru_cache.num_writable_bricks(),
        ));

        // write bricks to cache
        if !bricks.is_empty() {
            for (address, brick) in bricks {
                if brick.data.is_empty() {
                    self.page_table_directory.mark_as_empty(&address);
                } else {
                    // write brick to cache
                    let brick_location = self.lru_cache.add_cache_entry(
                        &brick.data,
                        input,
                    );

                    match brick_location {
                        Ok(brick_location) => {
                            // mark brick as mapped
                            self.page_table_directory
                                .mark_as_mapped(&address, &brick_location);
                        }
                        Err(_) => {
                            // todo: error handling
                            log::error!("Could not add brick to cache");
                        }
                    }
                }
                let brick_id = address.into();
                self.cached_bricks.insert(brick_id);
                self.requested_bricks.remove(&brick_id);
            }

            // update the page directory
            self.page_table_directory.commit_changes();
        }
    }

    /// Call this after rendering has completed to read back requests & usages
    pub fn update_cache(&mut self, input: &Input) {
        self.process_requests();
        self.process_new_bricks(input);
    }

    pub fn request_bricks(&mut self, brick_addresses: Vec<BrickAddress>) {
        if !brick_addresses.is_empty() {
            self.source.request_bricks(brick_addresses)
        }
    }
}

impl AsBindGroupEntries for SparseResidencyTexture3D {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self
                    .page_table_directory
                    .get_page_table_meta_as_binding_resource(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self
                    .page_table_directory
                    .get_page_directory_as_binding_resource(),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.lru_cache.get_cache_as_binding_resource(), //BindingResource::TextureView(&self.brick_cache.view),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.lru_cache.get_usage_buffer_as_binding_resource(), //BindingResource::TextureView(&self.brick_usage_buffer.view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&self.request_buffer.view),
            },
        ]
    }
}
