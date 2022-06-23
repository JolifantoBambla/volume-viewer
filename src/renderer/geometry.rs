use glam::{Vec2, Vec3A};

// stub for later, might need intersection, in-/outside tests, extend etc.
pub trait Bounds {}

#[readonly::make]
pub struct Bounds2D {
    pub min: Vec2,
    pub max: Vec2,
}

impl Bounds2D {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }
}

impl Bounds for Bounds2D {}

#[readonly::make]
pub struct Bounds3D {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Bounds3D {
    pub fn new(min: Vec3A, max: Vec3A) -> Self {
        Self { min, max }
    }
}

impl Bounds for Bounds3D {}
