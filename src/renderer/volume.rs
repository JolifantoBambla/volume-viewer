use wgpu::Extent3d;
use crate::renderer::geometry::Bounds3D;
use crate::renderer::resources::Texture;

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

// todo: this trait should abstract the volume's representation on the GPU
pub trait AccelerationStructure {
    /// Returns the axis aligned bounding box of this `AccelerationStructure`.
    fn bounds(&self) -> &AABB;
}

pub struct TrivialVolume {
    pub texture: Texture,
    pub bounds: AABB,
}

impl AccelerationStructure for TrivialVolume {
    fn bounds(&self) -> &AABB {
        &self.bounds
    }
}

pub struct Volume {
    acceleration_structure: Box<dyn AccelerationStructure>,

    // todo: move to struct containing numerical meta stuff?
    //maximum_value: f32,
}

impl Volume {
    pub fn bounds(&self) -> &AABB {
        self.acceleration_structure.bounds()
    }
}


// stub for later
pub struct MultiVolume {}
