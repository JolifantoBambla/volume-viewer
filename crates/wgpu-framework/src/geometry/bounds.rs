use glam::{Vec2, Vec3};
use std::ops::{Add, Mul, Sub};

pub trait Bounds {
    type VecN: Add<Output = Self::VecN> + Sub<Output = Self::VecN> + Mul<f32, Output = Self::VecN>;
    fn min(&self) -> Self::VecN;
    fn max(&self) -> Self::VecN;
    fn contains(&self, point: Self::VecN) -> bool;
    fn grow(&mut self, point: Self::VecN);
    fn corners(&self) -> Vec<Self::VecN>;
    fn diagonal(&self) -> Self::VecN {
        self.max() - self.min()
    }
    fn center(&self) -> Self::VecN {
        self.min() + self.diagonal() * 0.5
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Bounds2 {
    min: Vec2,
    max: Vec2,
}

impl Bounds2 {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self {
            min: min.min(max),
            max: max.max(min),
        }
    }
}

impl Bounds for Bounds2 {
    type VecN = Vec2;

    fn min(&self) -> Self::VecN {
        self.min
    }
    fn max(&self) -> Self::VecN {
        self.max
    }
    fn contains(&self, point: Self::VecN) -> bool {
        self.min.cmpge(point).all() && self.max.cmple(point).all()
    }
    fn grow(&mut self, point: Self::VecN) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }
    fn corners(&self) -> Vec<Self::VecN> {
        vec![
            self.min(),
            Vec2::new(self.min.x, self.max.y),
            Vec2::new(self.max.x, self.min.y),
            self.max(),
        ]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Bounds3 {
    min: Vec3,
    max: Vec3,
}

impl Bounds3 {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            min: min.min(max),
            max: max.max(min),
        }
    }

    pub fn set_xy(&mut self, other: Bounds2) {
        self.min = other.min.extend(self.min.z);
        self.max = other.max.extend(self.max.z);
    }
}

impl From<Vec3> for Bounds3 {
    fn from(v: Vec3) -> Self {
        Self::new(v, v)
    }
}

impl From<&[Vec3]> for Bounds3 {
    fn from(points: &[Vec3]) -> Self {
        let mut aabb = if let Some(p) = points.first() {
            Bounds3::from(*p)
        } else {
            Bounds3::from(Vec3::ZERO)
        };
        for p in points {
            aabb.grow(*p)
        }
        aabb
    }
}

impl Bounds for Bounds3 {
    type VecN = Vec3;
    fn min(&self) -> Vec3 {
        self.min
    }
    fn max(&self) -> Vec3 {
        self.max
    }
    fn contains(&self, point: Self::VecN) -> bool {
        self.min.cmpge(point).all() && self.max.cmple(point).all()
    }
    fn grow(&mut self, point: Self::VecN) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }
    fn corners(&self) -> Vec<Self::VecN> {
        vec![
            self.min(),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            self.max(),
        ]
    }
}
