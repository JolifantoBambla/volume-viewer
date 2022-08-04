use crate::util::extent::box_volume;
use glam::{UVec3, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[readonly::make]
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeResolutionMeta {
    /// The size of the volume in voxels.
    /// It is not necessarily a multiple of `brick_size`.
    /// See `padded_volume_size`.
    #[serde(rename = "volumeSize")]
    pub(crate) volume_size: [u32; 3],

    /// The size of the volume in voxels padded s.t. it is a multiple of `PageTableMeta::brick_size`.
    #[serde(rename = "paddedVolumeSize")]
    padded_volume_size: [u32; 3],

    /// The spatial extent of the volume.
    scale: [f32; 3],
}

// todo: move out of here
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultiResolutionVolumeMeta {
    /// The size of a brick in the brick cache. This is constant across all resolutions of the
    /// bricked multi-resolution volume.
    #[serde(rename = "brickSize")]
    pub(crate) brick_size: [u32; 3],

    /// The resolutions
    pub(crate) resolutions: Vec<VolumeResolutionMeta>,
}

impl MultiResolutionVolumeMeta {
    pub fn bricks_per_dimension(&self, level: usize) -> UVec3 {
        UVec3::from(self.resolutions[level].padded_volume_size) / UVec3::from(self.brick_size)
    }

    pub fn number_of_bricks(&self, level: usize) -> u32 {
        box_volume(self.bricks_per_dimension(level))
    }
}
