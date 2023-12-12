pub mod brick_cache_update;
mod cache_management;
mod page_table;

use glam::{UVec3, Vec3};
use std::cmp::min;
use std::collections::HashSet;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferUsages,
    CommandEncoder, Extent3d,
};
use wgsl_preprocessor::WGSLPreprocessor;

use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::Texture;
use crate::volume::{BrickAddress, VolumeDataSource};
use wgpu_framework::input::Input;

use crate::resource::sparse_residency::texture3d::brick_cache_update::CacheUpdateMeta;
use crate::resource::sparse_residency::texture3d::cache_management::lru::LRUCacheSettings;
use crate::resource::sparse_residency::texture3d::page_table::PageTableDirectory;
use crate::BrickedMultiResolutionMultiVolumeMeta;
use cache_management::{
    lru::LRUCache,
    process_requests::{ProcessRequests, Resources},
    Timestamp,
};
use wgpu_framework::context::Gpu;

pub struct SparseResidencyTexture3DOptions {
    pub max_visible_channels: u32,
    pub max_resolutions: u32,
    pub visible_channel_indices: Vec<u32>,
    // how many bricks can be requested per frame
    pub brick_request_limit: u32,
    // how many bricks can be uploaded per upload
    pub brick_transfer_limit: u32,
    pub cache_size: Extent3d,
    pub num_multi_buffering: u32,
    pub cache_time_to_live: u32,
}

impl Default for SparseResidencyTexture3DOptions {
    fn default() -> Self {
        Self {
            max_visible_channels: 16,
            max_resolutions: 16,
            visible_channel_indices: vec![0],
            brick_request_limit: 32,
            brick_transfer_limit: 32,
            cache_size: Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 1024,
            },
            num_multi_buffering: 2,
            cache_time_to_live: 300,
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
#[derive(Debug)]
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
        self.page_table_to_data_set
            .iter()
            .position(|&i| i == channel_index)
    }

    pub fn page_table_to_dataset(&self, channel_index: u32) -> u32 {
        self.page_table_to_data_set[channel_index as usize]
    }

    pub fn map_channel_indices(&self, channel_indices: &[u32]) -> Vec<Option<usize>> {
        channel_indices
            .iter()
            .map(|&i| self.dataset_to_page_table(i))
            .collect()
    }
}

impl Default for ChannelConfigurationState {
    /// By default only the first channel in the data set is visible.
    /// It is the only channel represented in the page table.
    fn default() -> Self {
        Self {
            page_table_to_data_set: vec![0],
            represented_channels: HashSet::from([0]),
            created_at: 0,
        }
    }
}

/// Manages a 3D sparse residency texture.
/// A sparse residency texture is not necessarily present in GPU memory as a whole.
#[derive(Debug)]
pub struct VolumeManager {
    ctx: Arc<Gpu>,

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
    cached_bricks: HashSet<u64>,
    channel_configuration: Vec<ChannelConfigurationState>,
}

impl VolumeManager {
    pub fn new(
        source: Box<dyn VolumeDataSource>,
        settings: SparseResidencyTexture3DOptions,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<Gpu>,
    ) -> Self {
        let timestamp = Timestamp::default();
        let timestamp_uniform_buffer = ctx.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&timestamp),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let volume_meta = source.meta();
        let brick_size = volume_meta.brick_size;

        let page_table_directory = PageTableDirectory::new(
            volume_meta,
            settings.max_visible_channels,
            settings.max_resolutions,
            ctx,
        );

        let lru_cache = LRUCache::new(
            LRUCacheSettings {
                cache_size: settings.cache_size,
                cache_entry_size: brick_size,
                num_multi_buffering: settings.num_multi_buffering,
                time_to_live: settings.cache_time_to_live,
            },
            &timestamp_uniform_buffer,
            wgsl_preprocessor,
            ctx,
        );

        // 1:1 mapping, 1 timestamp per brick in multi-res volume
        let request_buffer = Texture::create_u32_storage_3d(
            "Request Buffer".to_string(),
            ctx.device(),
            ctx.queue(),
            page_table_directory.extent(),
        );

        let process_requests_pass =
            ProcessRequests::new(settings.brick_request_limit as u32, wgsl_preprocessor, ctx);
        let process_requests_bind_group = process_requests_pass.create_bind_group(Resources {
            page_table_meta: page_table_directory.page_table_meta_buffer(),
            request_buffer: &request_buffer,
            timestamp: &timestamp_uniform_buffer,
        });

        Self {
            ctx: ctx.clone(),
            source,
            brick_transfer_limit: settings.brick_transfer_limit as usize,
            brick_request_limit: settings.brick_request_limit as usize,
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
                0,
            )],
        }
    }

    pub fn normalized_volume_size(&self) -> Vec3 {
        self.page_table_directory.normalized_volume_size()
    }

    pub fn encode_cache_management(
        &self,
        command_encoder: &mut CommandEncoder,
        timestamp: u32
        ) {
        self.ctx.queue().write_buffer(
            &self.timestamp_uniform_buffer,
            0 as BufferAddress,
            bytemuck::bytes_of(&Timestamp::new(timestamp)),
        );

        self.lru_cache.encode_lru_update(
            command_encoder,
            timestamp,
            #[cfg(feature = "timestamp-query")]
            timestamp_query_helper,
        );

        // find requested
        self.process_requests_pass.encode(
            command_encoder,
            &self.process_requests_bind_group,
            &self.request_buffer.extent,
            #[cfg(feature = "timestamp-query")]
            timestamp_query_helper,
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

            // the first frame, all bricks are requested because the request buffer is initialized to 0
            // the same probably goes for the LRU cache
            // todo: either start with frame number = 1 or initialize these buffers to u32::MAX
            if timestamp == 0 {
                return;
            }

            /*
            if !requested_ids.is_empty() {
                log::info!("requested ids {:?}", requested_ids);
            }
            */

            let mut brick_addresses =
                Vec::with_capacity(min(requested_ids.len(), self.brick_request_limit));
            for local_brick_id in requested_ids {
                let brick_address = self.map_from_page_table(
                    self.page_table_directory
                        .brick_id_to_brick_address(local_brick_id),
                    Some(timestamp),
                );
                let global_brick_id = brick_address.into();
                if !self.cached_bricks.contains(&global_brick_id)
                    && self.requested_bricks.insert(global_brick_id)
                {
                    brick_addresses.push(brick_address);
                }
                if brick_addresses.len() >= self.brick_request_limit {
                    break;
                }
            }

            /*
            if !brick_addresses.is_empty() {
                log::info!("requested {:?}", brick_addresses);
            }
             */

            self.request_bricks(brick_addresses);
        }
    }

    pub fn process_new_bricks(&mut self, last_frame_number: u32) -> CacheUpdateMeta {
        let bricks = self.source.poll_bricks(min(
            self.brick_transfer_limit,
            self.lru_cache.num_writable_bricks(),
        ));

        let mut update_result = CacheUpdateMeta::default();

        // write bricks to cache
        if !bricks.is_empty() {
            for b in &bricks {
                let (global_address, brick) = (&b.0, &b.1);
                if let Some(local_address) = self.map_to_page_table(&global_address, None) {
                    let brick_id = (*global_address).into();
                    let local_brick_id = self
                        .page_table_directory
                        .brick_address_to_brick_id(&local_address);
                    if brick.is_empty() {
                        self.page_table_directory.mark_as_empty(&local_address);
                        update_result
                            .add_mapped_brick_id(local_brick_id, true);
                    } else {
                        // write brick to cache
                        let brick_location = self.lru_cache.add_cache_entry(&brick, last_frame_number);
                        match brick_location {
                            Ok(brick_location) => {
                                // mark brick as mapped
                                let unmapped_brick_address = self
                                    .page_table_directory
                                    .map_brick(&local_address, &brick_location);
                                self.cached_bricks.insert(brick_id);

                                //log::info!("brick address {:?}", local_address);
                                // todo: check if first time
                                let mapped_first_time = true;
                                update_result
                                    .add_mapped_brick_id(local_brick_id, mapped_first_time);

                                if let Some(unmapped_brick_local_address) = unmapped_brick_address {
                                    let unmapped_brick_global_address = self.map_from_page_table(
                                        unmapped_brick_local_address,
                                        None, //input.frame().number(),
                                    );
                                    let unmapped_brick_id = unmapped_brick_global_address.into();
                                    self.cached_bricks.remove(&unmapped_brick_id);

                                    let local_unmapped_brick_id = self
                                        .page_table_directory
                                        .brick_address_to_brick_id(&unmapped_brick_local_address);
                                    update_result.add_unmapped_brick_id(local_unmapped_brick_id);
                                }
                            }
                            Err(_) => {
                                // todo: error handling
                                log::error!("Could not add brick to cache");
                                update_result.add_unsuccessful_map_attempt_brick_id(local_brick_id);
                            }
                        }
                    }
                    self.requested_bricks.remove(&brick_id);
                } else {
                    let brick_id = (*global_address).into();
                    self.requested_bricks.remove(&brick_id);
                }
            }

            //log::info!("self cached bricks len {}", self.cached_bricks.len());

            // update the page directory
            self.page_table_directory.commit_changes();
        }

        update_result
    }

    /// Call this after rendering has completed to read back requests & usages
    pub fn update_cache_meta(&mut self, input: &Input) {
        self.process_requests();
        // update CPU local LRU cache
        self.lru_cache.update_local_lru(input.frame().number());
    }


    pub fn request_bricks(&mut self, brick_addresses: Vec<BrickAddress>) {
        if !brick_addresses.is_empty() {
            self.source.request_bricks(brick_addresses)
        }
    }

    fn map_to_page_table(
        &self,
        brick_address: &BrickAddress,
        timestamp: Option<u32>,
    ) -> Option<BrickAddress> {
        let channel_configuration = if let Some(timestamp) = timestamp {
            self.get_channel_configuration(timestamp)
        } else {
            self.channel_configuration.last().unwrap()
        };
        channel_configuration
            .dataset_to_page_table(brick_address.channel)
            .map(|channel| {
                BrickAddress::new(brick_address.index, channel as u32, brick_address.level)
            })
    }

    fn map_from_page_table(&self, brick_address: BrickAddress, timestamp: Option<u32>) -> BrickAddress {
        let channel_configuration = if let Some(timestamp) = timestamp {
            self.get_channel_configuration(timestamp)
        } else {
            self.channel_configuration.last().unwrap()
        };
        BrickAddress::new(
            brick_address.index,
            channel_configuration.page_table_to_dataset(brick_address.channel),
            brick_address.level,
        )
    }

    ///
    ///
    /// # Arguments
    ///
    /// * `visible_channel_indices`: The list of indices of channels in the data set that are to be rendered.
    /// * `timestamp`: The timestamp at which the created channel configuration is active.
    pub fn add_channel_configuration(
        &mut self,
        selected_channel_indices: &Vec<u32>,
        timestamp: u32,
    ) -> Vec<Option<usize>> {
        let last_configuration = self.channel_configuration.last().unwrap();
        let selected_channels = HashSet::from_iter(selected_channel_indices.iter().copied());

        let mut new_selected: Vec<u32> = selected_channels
            .difference(&last_configuration.represented_channels)
            .copied()
            .collect();
        let no_longer_selected: Vec<u32> = last_configuration
            .represented_channels
            .difference(&selected_channels)
            .copied()
            .collect();

        let mut new_pt2d = last_configuration.page_table_to_data_set.clone();
        let new_num_channels = selected_channel_indices.len() + no_longer_selected.len();
        let channel_capacity = self.page_table_directory.channel_capacity();
        if new_num_channels <= channel_capacity {
            new_pt2d.append(&mut new_selected);
        } else {
            for i in 0..new_selected.len() {
                if i <= no_longer_selected.len() {
                    let idx = last_configuration
                        .dataset_to_page_table(no_longer_selected[i])
                        .unwrap();
                    new_pt2d[idx] = new_selected[i];
                    for local_brick_address in self.page_table_directory
                        .invalidate_channel_page_tables(idx as u32) {
                        let unmapped_brick_global_address = self.map_from_page_table(
                            local_brick_address,
                            None,
                        );
                        // todo: these need to be passed to the octree as well
                        self.cached_bricks.remove(&unmapped_brick_global_address.into());
                    }
                } else {
                    new_pt2d.push(new_selected[i]);
                }
            }
            self.page_table_directory.commit_changes()
        }
        self.channel_configuration
            .push(ChannelConfigurationState::from_mapping(new_pt2d, timestamp));
        self.channel_configuration
            .last()
            .unwrap()
            .map_channel_indices(selected_channel_indices.as_slice())
    }

    pub fn get_channel_configuration(&self, timestamp: u32) -> &ChannelConfigurationState {
        for c in self.channel_configuration.iter().rev() {
            if timestamp >= c.created_at {
                return c;
            }
        }
        panic!("SparseResidencyTexture3D has no channel configuration");
    }

    pub fn page_table_directory(&self) -> &PageTableDirectory {
        &self.page_table_directory
    }

    pub fn lru_cache(&self) -> &LRUCache {
        &self.lru_cache
    }

    pub fn brick_size(&self) -> UVec3 {
        self.lru_cache().cache_entry_size()
    }

    pub fn meta(&self) -> &BrickedMultiResolutionMultiVolumeMeta {
        self.source.meta()
    }

    pub fn brick_transfer_limit(&self) -> usize {
        self.brick_transfer_limit
    }

    pub fn source_mut(&mut self) -> &mut Box<dyn VolumeDataSource> {
        &mut self.source
    }
}

impl AsBindGroupEntries for VolumeManager {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self
                    .page_table_directory
                    .page_directory_meta_as_binding_resource(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self
                    .page_table_directory
                    .page_table_meta_as_binding_resource(),
            },
            BindGroupEntry {
                binding: 2,
                resource: self
                    .page_table_directory
                    .page_directory_as_binding_resource(),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.lru_cache.cache_as_binding_resource(), //BindingResource::TextureView(&self.brick_cache.view),
            },
            BindGroupEntry {
                binding: 4,
                resource: self.lru_cache.get_usage_buffer_as_binding_resource(), //BindingResource::TextureView(&self.brick_usage_buffer.view),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::TextureView(&self.request_buffer.view),
            },
        ]
    }
}
