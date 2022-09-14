pub mod data_source;
pub mod page_table;
pub mod volume_meta;

use crate::renderer::context::GPUContext;
use crate::renderer::pass::process_requests::{ProcessRequests, Resources, Timestamp};
use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::renderer::resources::Texture;
use crate::sparse_residency_resource::texture3d::data_source::{
    Brick, SparseResidencyTexture3DSource,
};
use crate::sparse_residency_resource::texture3d::page_table::{
    PageDirectoryMeta, PageTableEntryFlag,
};
use crate::sparse_residency_resource::texture3d::volume_meta::BrickAddress;
use crate::util::extent::{
    extent_to_uvec, index_to_subscript, subscript_to_index, uvec_to_extent, uvec_to_origin,
    box_volume,
};
use glam::{UVec3, UVec4, Vec3, Vec4};
use std::cmp::min;
use std::collections::HashSet;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferUsages,
    CommandEncoder, Extent3d, SubmissionIndex,
};
use wgsl_preprocessor::WGSLPreprocessor;

/// Manages a 3D sparse residency texture.
/// A sparse residency texture is not necessarily present in GPU memory as a whole.
pub struct SparseResidencyTexture3D {
    ctx: Arc<GPUContext>,
    meta: PageDirectoryMeta,
    //local_brick_cache: HashMap<PageTableAddress, Brick>,
    source: Box<dyn SparseResidencyTexture3DSource>,
    brick_transfer_limit: usize,
    brick_request_limit: usize,

    local_page_directory: Vec<UVec4>,
    requested_bricks: HashSet<u32>,
    cached_bricks: HashSet<u32>,

    // GPU resources
    page_table_meta_buffer: Buffer,
    page_directory: Texture,
    brick_cache: Texture,
    brick_usage_buffer: Texture,
    request_buffer: Texture,

    // Process Request Helper
    timestamp_uniform_buffer: Buffer,
    process_requests_pass: ProcessRequests,
    process_requests_bind_group: BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResMeta {
    // note: brick size is just copied
    brick_size: UVec4,
    page_table_offset: UVec4,
    page_table_extent: UVec4,
    volume_size: UVec4,
}

impl SparseResidencyTexture3D {
    pub fn new(
        source: Box<dyn SparseResidencyTexture3DSource>,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let volume_meta = source.get_meta();
        let meta = PageDirectoryMeta::new(volume_meta);

        let res_meta_data: Vec<ResMeta> = meta
            .resolutions
            .iter()
            .map(|r| ResMeta {
                brick_size: meta.brick_size.extend(0),
                page_table_offset: r.offset.extend(0),
                page_table_extent: r.extent.extend(0),
                volume_size: UVec3::from_slice(r.volume_meta.volume_size.as_slice()).extend(0),
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

        // todo: make configurable
        let brick_cache = Texture::create_brick_cache(
            &ctx.device,
            Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 1024,
            },
        );

        let brick_cache_size = extent_to_uvec(&brick_cache.extent);
        let bricks_per_dimension = brick_cache_size / meta.brick_size;

        // 1:1 mapping, 1 timestamp per brick in cache
        let brick_usage_buffer = Texture::create_u32_storage_3d(
            "Usage Buffer".to_string(),
            &ctx.device,
            &ctx.queue,
            uvec_to_extent(&bricks_per_dimension),
        );

        // 1:1 mapping, 1 timestamp per brick in multi-res volume
        let request_buffer = Texture::create_u32_storage_3d(
            "Request Buffer".to_string(),
            &ctx.device,
            &ctx.queue,
            page_directory.extent,
        );

        let brick_request_limit = 32;
        let timestamp = Timestamp::default();
        let timestamp_uniform_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&timestamp),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let process_requests_pass =
            ProcessRequests::new(brick_request_limit as u32, wgsl_preprocessor, ctx);
        let process_requests_bind_group = process_requests_pass.create_bind_group(Resources {
            page_table_meta: &page_table_meta_buffer,
            request_buffer: &request_buffer,
            timestamp: &timestamp_uniform_buffer,
        });

        Self {
            ctx: ctx.clone(),
            meta,
            source,
            brick_transfer_limit: 32,
            brick_request_limit,
            page_table_meta_buffer,
            page_directory,
            brick_cache,
            brick_usage_buffer,
            request_buffer,
            timestamp_uniform_buffer,
            process_requests_pass,
            process_requests_bind_group,
            local_page_directory,
            requested_bricks: HashSet::new(),
            cached_bricks: HashSet::new(),
        }
    }

    // todo: this should come from meta data (my current data set doesn't have such meta data)
    pub fn volume_scale(&self) -> Vec3 {
        let size = Vec3::new(
            self.meta.resolutions[0].volume_meta.volume_size[0] as f32,
            self.meta.resolutions[0].volume_meta.volume_size[1] as f32,
            self.meta.resolutions[0].volume_meta.volume_size[2] as f32,
        );
        size * self.meta.scale
    }

    pub fn encode_cache_management(&self, command_encoder: &mut CommandEncoder, timestamp: u32) {
        self.ctx.queue.write_buffer(
            &self.timestamp_uniform_buffer,
            0 as BufferAddress,
            bytemuck::bytes_of(&Timestamp::new(timestamp)),
        );

        // todo: find unused

        // find requested
        self.process_requests_pass.encode(
            command_encoder,
            &self.process_requests_bind_group,
            &self.request_buffer.extent,
        );
        self.process_requests_pass
            .encode_copy_result_to_readable(command_encoder);
    }

    /// Call this after rendering has completed to read back requests & usages
    pub fn update_cache(&mut self, _submission_index: SubmissionIndex, _temp_frame: u32) {
        // todo: map buffers
        self.process_requests_pass.map_for_reading();

        // todo: read and unmap buffers
        let requested_ids = self.process_requests_pass.read();

        if !requested_ids.is_empty() {
            // let requested_brick_addresses = requested_ids.iter().map(|id| id.to_be_bytes()).collect::<Vec<[u8; 4]>>();
            // log::info!("ids: {:?}", requested_brick_addresses);
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

        // todo: get actual free slots
        let free_slots: Vec<usize> = vec![0; 32];

        // todo: step by step:
        //  - insert bricks (for now just write them into their actual locations)
        //  - trace them
        //  - manage free slots (and all the rest

        let new_bricks = self
            .source
            .poll_bricks(min(self.brick_transfer_limit, free_slots.len()));
        self.add_bricks(new_bricks);

        // filter requested not in new_bricks
        // request
        // update cache
    }

    fn add_bricks(&mut self, bricks: Vec<(BrickAddress, Brick)>) {
        //log::info!("got bricks {}", bricks.len());
        if !bricks.is_empty() {
            // write each brick to the cache
            for (address, brick) in bricks {
                // todo: use free locations instead (should be a function param)
                let level = address.level as usize;
                let resolution = self.meta.resolutions[level];
                let offset = resolution.offset;
                let extent = resolution.extent;
                let location = UVec3::from_slice(address.index.as_slice());
                let brick_location = (offset + location) * self.meta.brick_size;
                let brick_extent = index_to_subscript(
                    (brick.data.len() as u32) - 1,
                    &uvec_to_extent(&self.meta.brick_size),
                ) + UVec3::ONE;


                log::info!("writing subregion\n  origin: {:?}, brick_size: {:?},\n  offset: {:?}, extent: {:?},\n  address: {:?}", brick_location, brick_extent, offset, extent, address.index);

                log::info!("brick data size {:?}, expected: {:?}", brick.data.len(), box_volume(&brick_extent));

                self.brick_cache.write_subregion(
                    brick.data.as_slice(),
                    uvec_to_origin(&brick_location),
                    uvec_to_extent(&brick_extent),
                    &self.ctx,
                );

                // mark brick as mapped
                let page_index =
                    subscript_to_index(&(offset + location), &self.page_directory.extent) as usize;
                self.local_page_directory[page_index] =
                    brick_location.extend(PageTableEntryFlag::Mapped as u32);

                let brick_id = address.into();
                self.cached_bricks.insert(brick_id);
                self.requested_bricks.remove(&brick_id);

                log::info!(
                    "brick size: {:?}, received size: {:?}",
                    self.meta.brick_size,
                    brick_extent
                );
            }

            // update the page directory
            self.page_directory
                .write(self.local_page_directory.as_slice(), &self.ctx);
        }
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
                resource: self.page_table_meta_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&self.page_directory.view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&self.brick_cache.view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&self.brick_usage_buffer.view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&self.request_buffer.view),
            },
        ]
    }
}
