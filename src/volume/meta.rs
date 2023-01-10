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

impl ResolutionMeta {
    pub fn volume_size_f32(&self) -> Vec3 {
        Vec3::new(self.volume_size.x as f32, self.volume_size.y as f32, self.volume_size.z as f32)
    }

    pub fn normalized_scale(&self) -> Vec3 {
        self.scale / self.scale.max_element()
    }

    pub fn volume_scale(&self) -> Vec3 {
        self.volume_size_f32() * self.normalized_scale()
    }
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

    pub fn top_level_volume_scale(&self) -> Vec3 {
        self.resolutions.first().map(|r| r.volume_scale()).unwrap_or(Vec3::ONE)
    }
}
