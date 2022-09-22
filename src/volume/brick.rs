use crate::util::extent::box_volume;
use glam::UVec3;
use serde::{Deserialize, Serialize};

#[readonly::make]
#[derive(Deserialize, Serialize)]
pub struct Brick {
    pub data: Vec<u8>,
    pub min: u8,
    pub max: u8,
}

#[readonly::make]
#[derive(Deserialize, Serialize)]
pub struct BrickAddress {
    /// x,y,z
    pub index: [u32; 3],
    pub level: u32,
    pub channel: u32,
}

impl From<u32> for BrickAddress {
    fn from(id: u32) -> Self {
        // todo: find out why these are in big endian - my system is little endian AND webgpu ensures little endian
        let bytes: [u8; 4] = id.to_be_bytes();
        Self {
            index: [bytes[0] as u32, bytes[1] as u32, bytes[2] as u32],
            level: bytes[3] as u32,
            // todo: figure out how to handle channels
            channel: 0,
        }
    }
}

impl From<BrickAddress> for u32 {
    fn from(brick_address: BrickAddress) -> Self {
        // todo: figure out how to handle channels
        (brick_address.index[0] << 24)
            + (brick_address.index[1] << 16)
            + (brick_address.index[2] << 8)
            + brick_address.level
    }
}
