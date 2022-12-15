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
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct BrickAddress {
    /// x,y,z
    // todo: refactor into x,y,z
    // todo: change type to u16
    pub index: UVec3,
    // todo: change type to u8
    pub channel: u32,
    // todo: rename to `resolution` (requires changes in JS code as well)
    // todo: change type to u8
    pub level: u32,
}

impl BrickAddress {
    pub fn new(index: UVec3, channel: u32, resolution: u32) -> Self {
        Self {
            index,
            channel,
            level: resolution,
        }
    }
}

impl From<BrickAddress> for u64 {
    /// A `BrickAddress` can be represented as a `u64`.
    /// Each spatial coordinate
    ///
    /// # Arguments
    ///
    /// * `brick_address`:
    ///
    /// returns: u64
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    fn from(brick_address: BrickAddress) -> Self {
        ((brick_address.index.x as u64) << 48)
            + ((brick_address.index.y as u64) << 32)
            + ((brick_address.index.z as u64) << 16)
            + ((brick_address.channel as u64) << 8)
            + brick_address.level as u64
    }
}
