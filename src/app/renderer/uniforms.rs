use glam::{Mat3, Mat4, UVec3, Vec2, Vec3};
use crate::app::scene::camera::OrbitCamera;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[readonly::make]
pub struct TransformUniform {
    pub object_to_world: Mat4,
    pub world_to_object: Mat4,
}

impl TransformUniform {
    pub fn from_object_to_world(object_to_world: Mat4) -> Self {
        Self {
            object_to_world,
            world_to_object: object_to_world.inverse(),
        }
    }

    pub fn from_world_to_object(world_to_object: Mat4) -> Self {
        Self {
            object_to_world: world_to_object.inverse(),
            world_to_object,
        }
    }
}

impl Default for TransformUniform {
    fn default() -> Self {
        Self {
            object_to_world: Mat4::IDENTITY,
            world_to_object: Mat4::IDENTITY,
        }
    }
}

pub trait ToTransformUniform {
    fn to_transform_uniform(&self) -> TransformUniform;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub transform: TransformUniform,
    pub projection: Mat4,
    pub inverse_projection: Mat4,
    pub camera_type: u32,
    padding: UVec3,
}

impl CameraUniform {
    pub fn new(view: Mat4, projection: Mat4, camera_type: u32) -> Self {
        Self {
            transform: TransformUniform::from_object_to_world(view),
            projection,
            inverse_projection: projection.inverse(),
            camera_type,
            padding: UVec3::new(0, 0, 0),
        }
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        CameraUniform::new(Mat4::IDENTITY, Mat4::IDENTITY, 0)
    }
}

impl From<&OrbitCamera> for CameraUniform {
    fn from(value: &OrbitCamera) -> Self {
        Self::new(
            // todo: thats weird -> shader uses wrong matrix?
            value.view().inverse(),
            value.projection(),
            value.is_orthographic() as u32,
        )
    }
}
