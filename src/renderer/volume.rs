use glam::Vec3A;
use crate::renderer::resources::Texture;



pub struct AABB {
    pub min: Vec3A,
    pub max: Vec3A,
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
    maximum_value: f32,
}

impl Volume {
    pub fn bounds(&self) -> &AABB {
        self.acceleration_structure.bounds()
    }
}


// stub for later
pub struct MultiVolume {}
