use crate::util::vec_hash_map::VecHashMap;
use crate::volume::BrickAddress;
use std::collections::HashMap;

#[readonly::make]
#[derive(Clone, Debug)]
pub struct MappedBrick {
    pub global_address: BrickAddress,
    pub min: u8,
    pub max: u8,
}

impl MappedBrick {
    pub fn new(global_address: BrickAddress, min: u8, max: u8) -> Self {
        Self {
            global_address,
            min,
            max,
        }
    }
}

#[readonly::make]
#[derive(Copy, Clone, Debug)]
pub struct UnmappedBrick {
    pub global_address: BrickAddress,
}

impl UnmappedBrick {
    pub fn new(global_address: BrickAddress) -> Self {
        Self { global_address }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChannelBrickCacheUpdateResult {
    mapped_bricks: VecHashMap<u32, MappedBrick>,
    unmapped_bricks: VecHashMap<u32, UnmappedBrick>,
}

impl ChannelBrickCacheUpdateResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_mapped(&mut self, brick: MappedBrick) {
        self.mapped_bricks.insert(brick.global_address.level, brick);
    }

    pub fn add_unmapped(&mut self, brick: UnmappedBrick) {
        self.unmapped_bricks
            .insert(brick.global_address.level, brick);
    }

    pub fn mapped_bricks(&self) -> &VecHashMap<u32, MappedBrick> {
        &self.mapped_bricks
    }

    pub fn unmapped_bricks(&self) -> &VecHashMap<u32, UnmappedBrick> {
        &self.unmapped_bricks
    }
}

#[readonly::make]
#[derive(Clone, Debug)]
pub struct BrickCacheUpdateResult {
    pub updates: HashMap<u32, ChannelBrickCacheUpdateResult>,
}

impl BrickCacheUpdateResult {
    pub fn new(updates: HashMap<u32, ChannelBrickCacheUpdateResult>) -> Self {
        Self { updates }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CacheUpdateMeta {
    mapped_local_brick_ids: Vec<u32>,
    unmapped_local_brick_ids: Vec<u32>,
    mapped_first_time_local_brick_ids: Vec<u32>,
    unsuccessful_map_attempt_local_brick_ids: Vec<u32>,
}

impl CacheUpdateMeta {
    pub fn is_empty(&self) -> bool {
        self.mapped_local_brick_ids.is_empty() && self.unmapped_local_brick_ids.is_empty()
    }
    pub fn add_mapped_brick_id(&mut self, brick_id: u32, new: bool) {
        self.mapped_local_brick_ids.push(brick_id);
        if new {
            self.mapped_first_time_local_brick_ids.push(brick_id);
        }
    }
    pub fn add_unmapped_brick_id(&mut self, brick_id: u32) {
        self.unmapped_local_brick_ids.push(brick_id);
    }
    pub fn add_unsuccessful_map_attempt_brick_id(&mut self, brick_id: u32) {
        self.unsuccessful_map_attempt_local_brick_ids.push(brick_id);
    }
    pub fn mapped_local_brick_ids(&self) -> &Vec<u32> {
        &self.mapped_local_brick_ids
    }
    pub fn unmapped_local_brick_ids(&self) -> &Vec<u32> {
        &self.unmapped_local_brick_ids
    }
    pub fn mapped_first_time_local_brick_ids(&self) -> &Vec<u32> {
        &self.mapped_first_time_local_brick_ids
    }
    pub fn unsuccessful_map_attempt_local_brick_ids(&self) -> &Vec<u32> {
        &self.unsuccessful_map_attempt_local_brick_ids
    }
}
