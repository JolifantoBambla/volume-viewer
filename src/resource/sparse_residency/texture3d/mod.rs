mod cache_management;
mod page_table;

use glam::{UVec3, UVec4, Vec3};
use std::cmp::min;
use std::collections::HashSet;
use std::sync::Arc;
use web_sys::console::time;
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
use crate::resource::sparse_residency::texture3d::cache_management::lru::LRUCacheSettings;

pub struct SparseResidencyTexture3DOptions {
    pub max_visible_channels: u32,
    pub max_resolutions: u32,
    pub visible_channel_indices: Vec<u32>,
    pub brick_request_limit: u32,
    pub cache_size: Extent3d,
    pub num_multi_buffering: u32,
    pub cache_time_to_live: u32,
}

impl Default for SparseResidencyTexture3DOptions {
    fn default() -> Self {
        Self {
            max_visible_channels: 17,
            max_resolutions: 15,
            visible_channel_indices: vec![0],
            brick_request_limit: 32,
            cache_size: Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 1024,
            },
            num_multi_buffering: 3,
            cache_time_to_live: u32::MAX,
        }
    }
}

///
/// C_d  ... number of channels in dataset
/// C_pt ... number of channels in page table
/// C_v  ... number of visible channels
/// C_d >= C_pt >= C_v
///
/// c_d  ... channel index in dataset in [0; C_d]
/// c_pt ... channel index in page table in [0; C_pt]
/// c_v  ... channel index in visible channels in [0; C_v]
/// Need to map:
///  - c_v  -> c_pt on GPU
///  - c_pt -> c_d  on CPU
pub struct ChannelConfigurationState {
    /// Maps from a channel in the
    /// Number of visible channels: channel_mapping.len()
    /// todo: only store pt2d here
    page_table_to_data_set: Vec<u32>,
    represented_channels: HashSet<u32>,
    created_at: u32,
}

impl ChannelConfigurationState {
    pub fn from_mapping(page_table_to_data_set: Vec<u32>, timestamp: u32) -> Self {
        let represented_channels = HashSet::from_iter(page_table_to_data_set.iter().copied());
        Self {
            page_table_to_data_set,
            represented_channels,
            created_at: timestamp,
        }
    }

    pub fn dataset_to_page_table(&self, channel_index: u32) -> Option<usize> {
        self.page_table_to_data_set.iter().position(|&i| i == channel_index)
    }

    pub fn page_table_to_dataset(&self, channel_index: u32) -> u32 {
        self.page_table_to_data_set[channel_index as usize]
    }

    pub fn map_channel_indices(&self, channel_indices: &Vec<u32>) -> Vec<Option<usize>> {
        channel_indices.iter().map(|&i| self.dataset_to_page_table(i)).collect()
    }
}

impl Default for ChannelConfigurationState {
    /// By default only the first channel in the data set is visible.
    /// It is the only channel represented in the page table.
    fn default() -> Self {
        Self {
            page_table_to_data_set: vec![0],
            represented_channels: HashSet::from([0]),
            created_at: 0
        }
    }
}

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

    // state tracking
    requested_bricks: HashSet<u64>,
    // todo: needs to be updated when cache entries are overridden
    cached_bricks: HashSet<u64>,
    channel_configuration: Vec<ChannelConfigurationState>,
}

impl SparseResidencyTexture3D {
    pub fn new(
        source: Box<dyn VolumeDataSource>,
        settings: SparseResidencyTexture3DOptions,
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

        let page_table_directory = PageTableDirectory::new(
            volume_meta,
            settings.max_visible_channels,
            settings.max_resolutions,
            ctx
        );

        let lru_cache = LRUCache::new(
            LRUCacheSettings {
                cache_size: settings.cache_size,
                cache_entry_size: brick_size,
                num_multi_buffering: settings.num_multi_buffering,
                time_to_live: settings.cache_time_to_live,
            },&timestamp_uniform_buffer,
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
            channel_configuration: vec![ChannelConfigurationState::from_mapping(
                settings.visible_channel_indices,
                0
            )],
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
        let brick_requests = self.process_requests_pass.read();

        // request bricks from data source
        if let Some(request) = brick_requests {
            let (requested_ids, timestamp) = request.into();
            let mut brick_addresses =
                Vec::with_capacity(min(requested_ids.len(), self.brick_request_limit));
            for id in requested_ids {
                let brick_address =
                    self.map_from_page_table(
                        self.page_table_directory.page_index_to_address(id),
                        timestamp
                    );
                let brick_id = brick_address.into();
                if !self.cached_bricks.contains(&brick_id) && self.requested_bricks.insert(brick_id) {
                    brick_addresses.push(brick_address);
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
                if let Some(local_address) = self.map_to_page_table(&address, None) {
                    if brick.data.is_empty() {
                        self.page_table_directory.mark_as_empty(&local_address);
                    } else {
                        // write brick to cache
                        let brick_location = self.lru_cache.add_cache_entry(&brick.data, input);

                        match brick_location {
                            Ok(brick_location) => {
                                // mark brick as mapped
                                self.page_table_directory
                                    .mark_as_mapped(&local_address, &brick_location);
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
                } else {
                    let brick_id = address.into();
                    self.requested_bricks.remove(&brick_id);
                }
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

    fn map_to_page_table(&self, brick_address: &BrickAddress, timestamp: Option<u32>) -> Option<BrickAddress> {
        let channel_configuration = if let Some(timestamp) = timestamp {
            self.get_channel_configuration(timestamp)
        } else {
            self.channel_configuration.last().unwrap()
        };
        if let Some(channel) = channel_configuration
            .dataset_to_page_table(brick_address.channel) {
            Some(BrickAddress::new(brick_address.index, channel as u32, brick_address.level))
        } else {
            None
        }
    }

    fn map_from_page_table(&self, brick_address: BrickAddress, timestamp: u32) -> BrickAddress {
        BrickAddress::new(
            brick_address.index,
            self.get_channel_configuration(timestamp)
                .page_table_to_dataset(brick_address.channel),
            brick_address.level,
        )
    }

    ///
    ///
    /// # Arguments
    ///
    /// * `visible_channel_indices`: The list of indices of channels in the data set that are to be rendered.
    /// * `timestamp`: The timestamp at which the created channel configuration is active.
    ///
    /// returns: ()
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn add_channel_configuration(&mut self, selected_channel_indices: &Vec<u32>, timestamp: u32) -> Vec<Option<usize>> {
        let last_configuration = self.channel_configuration.last().unwrap();
        let selected_channels = HashSet::from_iter(selected_channel_indices.iter().copied());

        let mut new_selected: Vec<u32> = selected_channels
            .difference(&last_configuration.represented_channels)
            .copied()
            .collect();
        let no_longer_selected: Vec<u32> = last_configuration.represented_channels
            .difference(&selected_channels)
            .copied()
            .collect();

        let mut new_pt2d = last_configuration.page_table_to_data_set.clone();
        let new_num_channels = selected_channel_indices.len() + no_longer_selected.len();
        let channel_capacity = self.page_table_directory.channel_capacity();
        if new_num_channels <= channel_capacity as usize {
            new_pt2d.append(&mut new_selected);
        } else {
            for i in 0..new_selected.len() {
                if i <= no_longer_selected.len() {
                    let idx = last_configuration
                        .dataset_to_page_table(no_longer_selected[i])
                        .unwrap();
                    new_pt2d[idx] = new_selected[i];
                    self.page_table_directory.invalidate_channel_page_tables(idx as u32);
                } else {
                    new_pt2d.push(new_selected[i]);
                }
            }
        }
        self.channel_configuration
            .push(ChannelConfigurationState::from_mapping(new_pt2d, timestamp));
        self.channel_configuration
            .last()
            .unwrap()
            .map_channel_indices(selected_channel_indices)
    }

    pub fn get_channel_configuration(&self, timestamp: u32) -> &ChannelConfigurationState {
        for c in self.channel_configuration.iter().rev() {
            if timestamp >= c.created_at {
                return c
            }
        }
        panic!("SparseResidencyTexture3D has no channel configuration");
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
