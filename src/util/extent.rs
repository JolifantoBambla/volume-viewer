use glam::UVec3;
use wgpu::Extent3d;

pub fn extent_to_uvec(extent: Extent3d) -> UVec3 {
    UVec3::new(extent.width, extent.height, extent.depth_or_array_layers)
}

pub fn uvec_to_extent(uvec: UVec3) -> Extent3d {
    Extent3d {
        width: uvec.x,
        height: uvec.y,
        depth_or_array_layers: uvec.z,
    }
}

pub fn box_volume(extent: UVec3) -> u32 {
    extent.into_array().iter().fold(1, |a, b| a * b)
}

pub fn extent_volume(extent: Extent3d) -> u32 {
    box_volume(extent_to_uvec(extent))
}
