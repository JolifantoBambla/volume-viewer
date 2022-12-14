use crate::util::extent::box_volume;
use glam::{UVec3, Vec3};
use serde::{Deserialize, Serialize};

#[readonly::make]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChannelInfo {
    pub name: String,
}

#[readonly::make]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolutionMeta {
    /// The size of the volume in voxels.
    /// It is not necessarily a multiple of `brick_size`.
    /// See `padded_volume_size`.
    #[serde(rename = "volumeSize")]
    pub volume_size: UVec3,

    /// The size of the volume in voxels padded s.t. it is a multiple of `PageTableMeta::brick_size`.
    #[serde(rename = "paddedVolumeSize")]
    pub padded_volume_size: UVec3,

    /// The spatial extent of the volume.
    pub scale: Vec3,
}

// todo: move out of here
#[readonly::make]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BrickedMultiResolutionMultiVolumeMeta {
    /// The size of a brick in the brick cache. This is constant across all resolutions of the
    /// bricked multi-resolution volume.
    #[serde(rename = "brickSize")]
    pub brick_size: UVec3,

    /// The spatial extent of the volume.
    pub scale: Vec3,

    pub channels: Vec<ChannelInfo>,

    /// The resolutions
    pub resolutions: Vec<ResolutionMeta>,
}

impl BrickedMultiResolutionMultiVolumeMeta {
    pub fn bricks_per_dimension(&self, level: usize) -> UVec3 {
        self.resolutions[level].padded_volume_size / self.brick_size
    }

    pub fn number_of_bricks(&self, level: usize) -> u32 {
        box_volume(&self.bricks_per_dimension(level))
    }
}
