use glam::Vec3A;

struct AABB {
    pub min: Vec3A,
    pub max: Vec3A,
}

trait AccelerationStructure {

}

struct TrivialVolume {

}

impl AccelerationStructure for TrivialVolume {

}

struct Volume {
    maximum_value: f32,
    bounds: AABB,
    acceleration_structure: Box<dyn AccelerationStructure>,
}

struct MultiVolume {

}
