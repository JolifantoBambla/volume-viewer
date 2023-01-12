use glam::{UVec2, UVec3, Vec3};
use wgpu::{Extent3d, Origin3d};

pub fn origin_to_uvec(origin: &Origin3d) -> UVec3 {
    UVec3::new(origin.x, origin.y, origin.z)
}

pub fn uvec_to_origin(uvec: &UVec3) -> Origin3d {
    Origin3d {
        x: uvec.x,
        y: uvec.y,
        z: uvec.z,
    }
}

pub fn extent_to_uvec(extent: &Extent3d) -> UVec3 {
    UVec3::new(extent.width, extent.height, extent.depth_or_array_layers)
}

pub fn uvec_to_extent(uvec: &UVec3) -> Extent3d {
    Extent3d {
        width: uvec.x,
        height: uvec.y,
        depth_or_array_layers: uvec.z,
    }
}

pub fn box_volume(extent: &UVec3) -> u32 {
    extent.to_array().iter().product()
}

pub fn extent_volume(extent: &Extent3d) -> u32 {
    box_volume(&extent_to_uvec(extent))
}

pub fn index_to_subscript(index: u32, extent: &Extent3d) -> UVec3 {
    let x = index % extent.width;
    let y = (index - x) / extent.width % extent.height;
    let z = ((index - x) / extent.width - y) / extent.height;
    UVec3::new(x, y, z)
}

pub fn subscript_to_index(subscript: &UVec3, extent: &Extent3d) -> u32 {
    subscript.x + extent.width * (subscript.y + extent.height * subscript.z)
}

pub trait IndexToSubscript {
    type Size;

    fn index_to_subscript(&self, index: u32) -> Self::Size;
}

pub trait SubscriptToIndex<Size = Self> {
    fn to_index(&self, size: &Size) -> u32;
}

pub trait ToNormalizedAddress<Size = Self> {
    type NormalizedAddress;

    fn to_normalized_address(&self, size: &Size) -> Self::NormalizedAddress;
}

pub trait ToSubscript {
    type Subscript;

    fn to_subscript(&self, size: Self::Subscript) -> Self::Subscript;
}

impl IndexToSubscript for UVec2 {
    type Size = Self;

    fn index_to_subscript(&self, index: u32) -> Self::Size {
        let x = index % self.x;
        let y = (index - x) / self.x % self.y;
        UVec2::new(x, y)
    }
}

impl SubscriptToIndex for UVec2 {
    fn to_index(&self, size: &Self) -> u32 {
        self.x + size.x * self.y
    }
}

impl IndexToSubscript for UVec3 {
    type Size = Self;

    fn index_to_subscript(&self, index: u32) -> Self::Size {
        let x = index % self.x;
        let y = (index - x) / self.x % self.y;
        let z = ((index - x) / self.x - y) / self.y;
        UVec3::new(x, y, z)
    }
}

impl SubscriptToIndex for UVec3 {
    fn to_index(&self, size: &Self) -> u32 {
        self.x + size.x * (self.y + size.y * self.z)
    }
}

impl ToNormalizedAddress for UVec3 {
    type NormalizedAddress = Vec3;

    fn to_normalized_address(&self, size: &Self) -> Self::NormalizedAddress {
        let self_f32 = Vec3::new(self.x as f32, self.y as f32, self.z as f32);
        let size_f32 = Vec3::new(size.x as f32, size.y as f32, size.z as f32);
        self_f32 / size_f32
    }
}

impl ToSubscript for Vec3 {
    type Subscript = UVec3;

    fn to_subscript(&self, size: Self::Subscript) -> Self::Subscript {
        let size_f32 = Vec3::new(size.x as f32, size.y as f32, size.z as f32);
        let subscript_f32 = *self * size_f32;
        UVec3::new(
            (subscript_f32.x.floor() as u32).min(size.x - 1),
            (subscript_f32.y.floor() as u32).min(size.y - 1),
            (subscript_f32.z.floor() as u32).min(size.z - 1),
        )
    }
}
