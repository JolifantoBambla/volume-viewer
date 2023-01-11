use crate::event::window::OnResize;
use crate::geometry::bounds::{Bounds, Bounds2, Bounds3};
use crate::scene::transform::{OrthonormalBasis, Transform, Transformable};
use glam::{Mat4, Vec2, Vec3};

#[derive(Copy, Clone, Debug)]
pub struct Camera {
    view: CameraView,
    projection: Projection,
}

impl Camera {
    pub fn new(view: CameraView, projection: Projection) -> Self {
        Self { view, projection }
    }
    pub fn view(&self) -> &CameraView {
        &self.view
    }
    pub fn view_mut(&mut self) -> &mut CameraView {
        &mut self.view
    }
    pub fn projection(&self) -> &Projection {
        &self.projection
    }
    pub fn projection_mut(&mut self) -> &mut Projection {
        &mut self.projection
    }
    pub fn set_projection(&mut self, projection: Projection) {
        self.projection = projection;
    }
    pub fn view_mat(&self) -> Mat4 {
        self.view.view()
    }
    pub fn inverse_view_mat(&self) -> Mat4 {
        self.view_mat().inverse()
    }
    pub fn projection_mat(&self) -> Mat4 {
        self.projection.projection()
    }
    pub fn inverse_projection_mat(&self) -> Mat4 {
        self.projection_mat().inverse()
    }
    pub fn zoom_in(&mut self, delta: f32) {
        self.view.zoom_in(delta);
    }

    pub fn zoom_out(&mut self, delta: f32) {
        self.zoom_in(-delta);
    }
}

impl Transformable for Camera {
    fn transform(&self) -> &Transform {
        self.view.transform()
    }

    fn transform_mut(&mut self) -> &mut Transform {
        self.view.transform_mut()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CameraView {
    transform: Transform,
    center_of_projection: Vec3,
}

impl CameraView {
    pub fn new(position: Vec3, center_of_projection: Vec3, up: Vec3) -> Self {
        Self {
            transform: Transform::new(
                position,
                OrthonormalBasis::new(center_of_projection - position, up),
                Vec3::ONE,
            ),
            center_of_projection,
        }
    }

    pub fn center_of_projection(&self) -> Vec3 {
        self.center_of_projection
    }

    pub fn set_center_of_projection(&mut self, center_of_projection: Vec3) {
        self.center_of_projection = center_of_projection;
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(
            self.transform.position(),
            self.center_of_projection,
            self.transform.up(),
        )
    }

    pub fn zoom_in(&mut self, delta: f32) {
        let distance = self
            .transform
            .position()
            .distance(self.center_of_projection);
        if distance > f32::EPSILON || delta < 0.0 {
            let movement = self.transform.forward()
                * if distance <= delta {
                    distance - f32::EPSILON
                } else {
                    delta
                };
            self.translate(movement);
        }
    }

    pub fn zoom_out(&mut self, delta: f32) {
        self.zoom_in(-delta);
    }
}

impl Default for CameraView {
    fn default() -> Self {
        Self::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }
}

impl Transformable for CameraView {
    fn transform(&self) -> &Transform {
        &self.transform
    }

    fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }
}

#[derive(Copy, Clone, Debug)]
pub struct OrthographicProjection {
    projection: Mat4,
    frustum: Bounds3,
}

impl OrthographicProjection {
    pub fn new(frustum: Bounds3) -> Self {
        let projection = Mat4::orthographic_rh(
            frustum.min().x,
            frustum.max().x,
            frustum.min().y,
            frustum.max().y,
            frustum.min().z,
            frustum.max().z,
        );
        Self {
            projection,
            frustum,
        }
    }

    pub fn projection(&self) -> Mat4 {
        self.projection
    }
    pub fn frustum(&self) -> &Bounds3 {
        &self.frustum
    }
}

impl OnResize for OrthographicProjection {
    fn on_resize(&mut self, width: u32, height: u32) {
        let width_half = (width / 2) as f32;
        let height_half = (height / 2) as f32;
        let xy_bounds = Bounds2::new(
            Vec2::new(-width_half, -height_half),
            Vec2::new(width_half, height_half),
        );
        self.frustum.set_xy(xy_bounds);
        self.projection = Mat4::orthographic_rh(
            self.frustum.min().x,
            self.frustum.max().x,
            self.frustum.min().y,
            self.frustum.max().y,
            self.frustum.min().z,
            self.frustum.max().z,
        );
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PerspectiveProjection {
    projection: Mat4,
    fov_y: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32,
}

impl PerspectiveProjection {
    pub fn new(fov_y: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let projection = Mat4::perspective_rh(fov_y, aspect_ratio, z_near, z_far);
        Self {
            projection,
            fov_y,
            aspect_ratio,
            z_near,
            z_far,
        }
    }
    fn update_projection(&mut self) {
        self.projection =
            Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.z_near, self.z_far);
    }

    pub fn projection(&self) -> Mat4 {
        self.projection
    }
    pub fn fov_y(&self) -> f32 {
        self.fov_y
    }
    pub fn fov_y_degrees(&self) -> f32 {
        self.fov_y.to_degrees()
    }
    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }
    pub fn z_near(&self) -> f32 {
        self.z_near
    }
    pub fn z_far(&self) -> f32 {
        self.z_far
    }
    pub fn set_fov_y(&mut self, fov_y: f32) {
        self.fov_y = fov_y;
        self.update_projection();
    }
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
        self.update_projection();
    }
    pub fn set_z_near(&mut self, z_near: f32) {
        self.z_near = z_near;
        self.update_projection();
    }
    pub fn set_z_far(&mut self, z_far: f32) {
        self.z_far = z_far;
        self.update_projection();
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Projection {
    Orthographic(OrthographicProjection),
    Perspective(PerspectiveProjection),
}

impl Projection {
    pub fn new_orthographic(frustum: Bounds3) -> Self {
        Self::Orthographic(OrthographicProjection::new(frustum))
    }
    pub fn new_perspective(fov_y: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        Self::Perspective(PerspectiveProjection::new(
            fov_y,
            aspect_ratio,
            z_near,
            z_far,
        ))
    }
    pub fn projection(&self) -> Mat4 {
        match self {
            Self::Orthographic(o) => o.projection(),
            Self::Perspective(p) => p.projection(),
        }
    }
}

impl OnResize for Projection {
    fn on_resize(&mut self, width: u32, height: u32) {
        match self {
            Self::Orthographic(o) => o.on_resize(width, height),
            Self::Perspective(p) => p.set_aspect_ratio(width as f32 / height as f32),
        }
    }
}
