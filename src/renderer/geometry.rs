use glam::{Vec2, Vec3};

// stub for later, might need intersection, in-/outside tests, extend etc.
pub trait Bounds {}

#[readonly::make]
#[derive(Clone)]
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
#[derive(Clone)]
pub struct Bounds3D {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds3D {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
}

impl Bounds for Bounds3D {}
