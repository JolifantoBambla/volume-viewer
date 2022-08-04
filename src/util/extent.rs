use glam::UVec3;
use wgpu::{Extent3d, Origin3d};

pub fn uvec_to_origin(uvec: UVec3) -> Origin3d {
    Origin3d {
        x: uvec.x,
        y: uvec.y,
        z: uvec.z,
    }
}

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
    extent.to_array().iter().fold(1, |a, b| a * b)
}

pub fn extent_volume(extent: Extent3d) -> u32 {
    box_volume(extent_to_uvec(extent))
}

pub fn index_to_subscript(index: u32, extent: Extent3d) -> UVec3 {
    let num_elements = extent_volume(extent);
    let page_index = index % num_elements;
    let x = index % extent.width;
    let y = (index - x) / extent.width % extent.height;
    let z =((index - x) / extent.width - y) / extent.height;
    UVec3::new(x, y, z)
}

pub fn subscript_to_index(subscript: UVec3, extent: Extent3d) -> u32 {
    subscript.x + extent.width * (subscript.y + extent.height * subscript.z)
}
