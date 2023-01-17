use glam::{Mat3, Mat4, Vec2, Vec3};
use wgpu_framework::geometry::bounds::{Bounds, Bounds2, Bounds3};

#[derive(Copy, Clone)]
pub struct CameraView {
    position: Vec3,
    center_of_projection: Vec3,
    up: Vec3,
}

impl CameraView {
    pub fn new(position: Vec3, center_of_projection: Vec3, up: Vec3) -> Self {
        Self {
            position,
            center_of_projection,
            up: up.normalize(),
        }
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn center_of_projection(&self) -> Vec3 {
        self.center_of_projection
    }

    pub fn set_center_of_projection(&mut self, center_of_projection: Vec3) {
        self.center_of_projection = center_of_projection;
    }

    pub fn up(&self) -> Vec3 {
        self.up
    }

    pub fn set_up(&mut self, up: Vec3) {
        self.up = up.normalize();
    }

    pub fn look_at(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.center_of_projection, self.up)
    }

    pub fn forward(&self) -> Vec3 {
        (self.center_of_projection - self.position).normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up)
    }

    pub fn translate(&mut self, translation: Vec3) {
        self.position += translation;
        self.center_of_projection += translation;
    }

    pub fn move_forward(&mut self, delta: f32) {
        // todo: ensure that center of projection is not behind camera
        self.position += self.forward() * delta;
    }

    pub fn move_backward(&mut self, delta: f32) {
        self.move_forward(-delta);
    }

    pub fn move_right(&mut self, delta: f32) {
        self.translate(self.right() * delta);
    }

    pub fn move_left(&mut self, delta: f32) {
        self.move_right(-delta);
    }

    pub fn move_up(&mut self, delta: f32) {
        self.translate(self.up * delta);
    }

    pub fn move_down(&mut self, delta: f32) {
        self.move_up(-delta);
    }

    pub fn orbit(&mut self, delta: Vec2, invert: bool) {
        if !(is_close_to_zero(delta.x) && is_close_to_zero(delta.y)) {
            let delta_scaled = delta * (std::f32::consts::PI * 2.);

            // choose origin to orbit around
            let origin = if invert {
                self.position
            } else {
                self.center_of_projection
            };

            // choose point that is being orbited
            let position = if invert {
                self.center_of_projection
            } else {
                self.position
            };

            let center_to_eye = position - origin;
            let radius = center_to_eye.length();

            let z = center_to_eye.normalize();
            let y = self.up;
            let x = y.cross(z).normalize();

            let y_rotation = Mat3::from_axis_angle(y, -delta_scaled.x);
            let x_rotation = Mat3::from_axis_angle(x, -delta_scaled.y);

            let rotated_y = y_rotation.mul_vec3(z);
            let rotated_x = x_rotation.mul_vec3(rotated_y);

            let new_position = origin
                + (if rotated_x.x.signum() == rotated_y.x.signum() {
                    rotated_x
                } else {
                    rotated_y
                } * radius);
            if invert {
                self.center_of_projection = new_position;
            } else {
                self.position = new_position;
            }
        }
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

#[derive(Copy, Clone)]
#[readonly::make]
pub struct Projection {
    projection: Mat4,
    is_orthographic: bool,
}

impl Projection {
    pub fn new_orthographic(frustum: Bounds3) -> Self {
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
            is_orthographic: true,
        }
    }

    pub fn new_perspective(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let projection = Mat4::perspective_rh(fov_y_radians, aspect_ratio, z_near, z_far);
        Self {
            projection,
            is_orthographic: false,
        }
    }

    pub fn projection(&self) -> Mat4 {
        self.projection
    }

    pub fn is_orthographic(&self) -> bool {
        self.is_orthographic
    }
}

#[derive(Copy, Clone)]
pub struct Camera {
    pub view: CameraView,
    projection: Projection,
}

impl Camera {
    pub fn new(view: CameraView, projection: Projection) -> Self {
        Self { view, projection }
    }

    pub fn view(&self) -> &CameraView {
        &self.view
    }

    pub fn set_view(&mut self, view: CameraView) {
        self.view = view;
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    pub fn set_projection(&mut self, projection: Projection) {
        self.projection = projection;
    }
}

pub fn compute_screen_to_raster(raster_resolution: Vec2, screen_size: Option<Bounds2>) -> Mat4 {
    let screen = if let Some(screen_size) = screen_size {
        screen_size
    } else {
        let aspect_ratio = raster_resolution.x / raster_resolution.y;
        if aspect_ratio > 1. {
            Bounds2::new(Vec2::new(-aspect_ratio, -1.0), Vec2::new(aspect_ratio, 1.0))
        } else {
            Bounds2::new(
                Vec2::new(-1.0, -1.0 / aspect_ratio),
                Vec2::new(1.0, 1.0 / aspect_ratio),
            )
        }
    };
    glam::Mat4::from_scale(raster_resolution.extend(1.0))
        .mul_mat4(&glam::Mat4::from_scale(glam::Vec3::new(
            1.0 / (screen.max().x - screen.min().x),
            1.0 / (screen.min().y - screen.max().y),
            1.0,
        )))
        .mul_mat4(&glam::Mat4::from_translation(glam::Vec3::new(
            -screen.min().x,
            -screen.max().y,
            0.,
        )))
}

pub fn compute_raster_to_screen(raster_resolution: Vec2, screen_size: Option<Bounds2>) -> Mat4 {
    compute_screen_to_raster(raster_resolution, screen_size).inverse()
}

// todo: move this to some util module
fn is_close_to_zero(val: f32) -> bool {
    val.abs() < f32::EPSILON
}
