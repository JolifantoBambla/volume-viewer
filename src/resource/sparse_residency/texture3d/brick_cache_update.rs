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
