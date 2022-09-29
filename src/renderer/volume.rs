use crate::renderer::geometry::Bounds3D;
use wgpu::Extent3d;

pub type AABB = Bounds3D;

#[readonly::make]
pub struct RawVolumeBlock {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub data: Vec<u8>,
    pub max: u32,
}

impl RawVolumeBlock {
    pub fn new(data: Vec<u8>, max: u32, width: u32, height: u32, depth: u32) -> Self {
        Self {
            data,
            width,
            height,
            depth,
            max,
        }
    }

    pub fn create_extent(&self) -> wgpu::Extent3d {
        Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: self.depth,
        }
    }

    pub fn create_vec3(&self) -> glam::Vec3 {
        glam::Vec3::new(self.width as f32, self.height as f32, self.depth as f32)
    }
}
