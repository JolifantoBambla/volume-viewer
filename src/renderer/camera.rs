use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use glam::{Mat3, Mat4, UVec3, Vec2, Vec3, Vec3A};
use crate::renderer::geometry::{Bounds2D, Bounds3D};

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
        CameraUniform::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            0
        )
    }
}

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

            let z = center_to_eye.normalize();;
            let y = self.up;
            let x = y.cross(z).normalize();

            let y_rotation = Mat3::from_axis_angle(y, -delta_scaled.x);
            let x_rotation = Mat3::from_axis_angle(x, -delta_scaled.y);

            let rotated_y = y_rotation.mul_vec3(z);
            let rotated_x = x_rotation.mul_vec3(rotated_y);

            let new_position = origin + (
                if rotated_x.x.signum() == rotated_y.x.signum() {
                    rotated_x
                } else {
                    rotated_y
                } * radius
            );
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

impl ToTransformUniform for CameraView {
    fn to_transform_uniform(&self) -> TransformUniform {
        // todo: might need to switch that around
        TransformUniform::from_world_to_object(self.look_at())
    }
}

#[derive(Clone)]
#[readonly::make]
pub struct Projection {
    projection: Mat4,
    is_orthographic: bool,
}

impl Projection {
    pub fn new_orthographic(frustum: Bounds3D) -> Self {
        let projection = Mat4::orthographic_rh(
            frustum.min.x, frustum.max.x,
            frustum.min.y, frustum.max.y,
            frustum.min.z, frustum.max.z
        );
        Self{
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

pub struct Camera {
    pub view: CameraView,
    projection: Projection,
}

impl Camera {
    pub fn new(view: CameraView, projection: Projection) -> Self {
        Self {
            view,
            projection,
        }
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

    pub fn create_uniform(&self) -> CameraUniform {
        CameraUniform::new(
            self.view.look_at().inverse(),
            self.projection.projection(),
            self.projection.is_orthographic() as u32
        )
    }
}

pub fn compute_screen_to_raster(raster_resolution: Vec2, screen_size: Option<Bounds2D>) -> Mat4 {
    let screen = if let Some(screen_size) = screen_size {
        screen_size
    } else {
        let aspect_ratio = raster_resolution.x / raster_resolution.y;
        if aspect_ratio > 1. {
            Bounds2D::new(Vec2::new(-aspect_ratio, -1.0),Vec2::new(aspect_ratio, 1.0))
        } else {
            Bounds2D::new(Vec2::new(-1.0, -1.0 / aspect_ratio), Vec2::new(1.0, 1.0 / aspect_ratio))
        }
    };
    glam::Mat4::from_scale(raster_resolution.extend(1.0))
        .mul_mat4(&glam::Mat4::from_scale(
            glam::Vec3::new(
                1.0 / (screen.max.x - screen.min.x),
                1.0 / (screen.min.y - screen.max.y),
                1.0
            )))
        .mul_mat4(&glam::Mat4::from_translation(
            glam::Vec3::new(
                -screen.min.x,
                -screen.max.y,
                0.
            )))
}

pub fn compute_raster_to_screen(raster_resolution: Vec2, screen_size: Option<Bounds2D>) -> Mat4 {
    compute_screen_to_raster(raster_resolution, screen_size).inverse()
}

fn is_close_to_zero(val: f32) -> bool {
    val.abs() < f32::EPSILON
}



pub trait Motion {
}

// todo: this is going to need refactoring at some point (took from my ray tracing Vulkan sample: https://github.com/JolifantoBambla/vk-samples/blob/main/samples/ray-tracing/ray-tracing.lisp)
#[derive(PartialEq)]
pub enum MotionMode {
    Examine,
    Walk,
    Fly,
    Trackball,
}

impl Display for MotionMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            MotionMode::Examine => "Examine",
            MotionMode::Walk => "Walk",
            MotionMode::Fly => "Fly",
            MotionMode::Trackball => "Trackball"
        })
    }
}

pub enum MotionAction {
    Orbit,
    Dolly,
    Pan,
    LookAround,
}

#[derive(PartialEq)]
pub enum MouseButton {
    Left,
    Right,
}

#[derive(PartialEq, Eq, Hash)]
pub enum Modifier {
    Alt,
    Ctrl,
    Shift,
}


pub struct CameraController {
    pub camera_position: Vec3,
    pub center_position: Vec3,
    pub up_vector: Vec3,
    pub roll: f32,
    pub matrix: Mat4,
    pub window_size: Vec2,
    pub movement_speed: f32,
    pub mouse_position: Vec2,
    pub mode: MotionMode,
}

impl Display for CameraController {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CameraController {{\
                camera_position: {},\
                center_position: {},\
                up_vector: {},\
                roll: {},\
                matrix: {},\
                window_size: {},\
                movement_speed: {},\
                mouse_position: {},\
                mode: {},\
            }}",
            self.camera_position,
            self.center_position,
            self.up_vector,
            self.roll,
            self.matrix,
            self.window_size,
            self.movement_speed,
            self.mouse_position,
            self.mode
        )
    }
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            camera_position: Vec3::new(10., 10., 10.),
            center_position: Vec3::new(0., 0., 0.),
            up_vector: Vec3::new(0., 1., 0.),
            roll: 0.,
            matrix: Mat4::IDENTITY,
            window_size: Vec2::new(1., 1.),
            movement_speed: 30.,
            mouse_position: Vec2::new(0., 0.),
            mode: MotionMode::Examine,
        }
    }
}


impl CameraController {
    /// Update the cameras view matrix
    pub fn update(&mut self) {
        self.matrix = Mat4::look_at_rh(
            self.camera_position,
            self.center_position,
            self.up_vector);
        if !(is_close_to_zero(self.roll)) {
            self.matrix *= Mat4::from_axis_angle(
                Vec3::new(0., 0., 1.),
                self.roll);
        }
        if self.matrix.is_nan() {
            log::error!("got NaN in cam controller: {}", self);
        }
    }
    
    pub fn set_look_at(&mut self, eye: Vec3, center: Vec3, up: Vec3) {
        self.camera_position = eye;
        self.center_position = center;
        self.up_vector = up;
        self.update();
    }

    fn orbit(&mut self, delta: Vec2, invert: bool) {
        log::info!("invert: {}", invert);
        if !(is_close_to_zero(delta.x) && is_close_to_zero(delta.y)) {
            let d = delta * (std::f32::consts::PI * 2.);
            let origin = if invert {
                self.camera_position
            } else {
                self.center_position
            };
            let position = if invert {
                self.center_position
            } else {
                self.camera_position
            };
            let mut center_to_eye = position - origin;
            let radius = center_to_eye.length();
            let z = center_to_eye.normalize();
            let y_rotation = Mat3::from_axis_angle(
                self.up_vector,
                -d.x);
            center_to_eye = y_rotation.mul_vec3(z);
            let x = self.up_vector.cross(z).normalize();
            let x_rotation = Mat3::from_axis_angle(x, -d.y);
            let rotated_vector = x_rotation.mul_vec3(center_to_eye);
            if rotated_vector.x.signum() == center_to_eye.x.signum() {
                center_to_eye = rotated_vector;
            }
            center_to_eye *= radius;
            let new_pos = center_to_eye + origin;
            if invert {
                self.center_position = new_pos;
            } else {
                self.camera_position = new_pos;
            }
        }
    }

    fn dolly(&mut self, delta: Vec2) {
        let mut z = self.center_position - self.camera_position;
        let mut distance = z.length();
        if !(is_close_to_zero(distance)) {
            log::info!("actually doing something");
            let dd = if self.mode == MotionMode::Examine {
                if delta.x.abs() > delta.y.abs() {
                    delta.x
                } else {
                    delta.y
                }
            } else {
                -delta.y
            };
            let mut factor = self.movement_speed * (dd / distance);
            distance /= 10.;
            distance = f32::max(distance, 0.001);
            factor *= distance;
            if 1.0 <= factor {
                z *= factor;
                if self.mode == MotionMode::Walk {
                    if self.up_vector.y > self.up_vector.z {
                        z.y = 0.;
                    } else {
                        z.z = 0.;
                    }
                }
                self.camera_position += z;
                if self.mode == MotionMode::Examine {
                    self.center_position += z;
                }
            }
        }
    }

    pub fn wheel(&mut self, value: f32) {
        let dx = ((value.abs() * value) / self.window_size.x as f32) * self.movement_speed;
        self.dolly(Vec2::new(dx, dx));
        self.update();
    }

    fn pan(&mut self, delta: Vec2) {
        let mut z = self.center_position - self.camera_position;
        let distance = z.length() / 0.785;
        z = z.normalize();
        let mut x = self.up_vector.cross(z).normalize();
        let mut y = z.cross(x).normalize();
        x *= distance * -delta.x;
        y *= distance * delta.y;
        if self.mode == MotionMode::Fly {
            x *= -1.;
            y *= -1.;
        }
        self.camera_position += x + y;
        self.center_position += x + y;
    }

    fn trackball(&mut self, position: Vec2) {
        const TRACKBALL_SIZE: f32 = 0.8;
        fn project_onto_tb_sphere(p: Vec2) -> f32 {
            let d = p.length();
            if d < TRACKBALL_SIZE * std::f32::consts::FRAC_1_SQRT_2 {
                f32::sqrt(TRACKBALL_SIZE.powf(2.) - d.powf(2.))
            } else {
                f32::exp2(TRACKBALL_SIZE / std::f32::consts::SQRT_2) / d
            }
        }
        let p0 = Vec2::new(
            2. * (self.mouse_position.x as f32 - (self.window_size.x as f32 / 2.)) / self.window_size.x as f32,
            2. * ((self.window_size.y as f32 / 2.) - self.mouse_position.y as f32) / self.window_size.y as f32,
        );
        let p1 = Vec2::new(
            2. * (position.x as f32 - (self.window_size.x as f32 / 2.)) / self.window_size.x as f32,
            2. * ((self.window_size.y as f32 / 2.) - position.y as f32) / self.window_size.y as f32,
        );
        let ptb0 = Vec3::new(p0.x, p0.y, project_onto_tb_sphere(p0));
        let ptb1 = Vec3::new(p1.x, p1.y, project_onto_tb_sphere(p1));
        let axis = ptb0.cross(ptb1).normalize();
        let mut s = ((ptb0 - ptb1) / (TRACKBALL_SIZE * 2.)).length();
        if s > 1. {
            s = 1.;
        } else if s < -1. {
            s = -1.;
        }
        let rad = s.asin() * 2.;
        let rot_axis = self.matrix.mul_vec4(axis.extend(0.)).truncate();
        let rot_mat = Mat3::from_axis_angle(rot_axis, rad);
        let pnt = self.camera_position - self.center_position;
        let pnt2 = rot_mat.mul_vec3(pnt);
        let up2 = rot_mat.mul_vec3(self.up_vector);
        self.camera_position = self.center_position + pnt2;
        self.up_vector = up2;
    }

    fn motion(&mut self, position: Vec2, action: MotionAction) {
        let delta = Vec2::new(
            (position.x as f32 - self.mouse_position.x) / self.window_size.x,
            (position.y as f32 - self.mouse_position.y) / self.window_size.y
        );
        match action {
            MotionAction::Orbit => {
                log::info!("orbit");
                self.orbit(delta, self.mode == MotionMode::Trackball);
            }
            MotionAction::Dolly => {
                log::info!("dolly");
                self.dolly(delta);
            }
            MotionAction::Pan => {
                log::info!("pan");
                self.pan(delta);
            }
            MotionAction::LookAround => {
                if self.mode == MotionMode::Trackball {
                    log::info!("trackball");
                    self.trackball(position);
                } else {
                    log::info!("look around: orbit");
                    self.orbit(Vec2::new(delta.x, -delta.y), true);
                }
            }
        }
        self.update();
        self.mouse_position = position;
    }

    pub fn mouse_move(&mut self, position: Vec2, mouse_button: MouseButton, modifiers: &HashSet<Modifier>) {
        let action = if mouse_button == MouseButton::Left {
            log::info!("left");
            if (modifiers.contains(&Modifier::Ctrl) && modifiers.contains(&Modifier::Shift)) || modifiers.contains(&Modifier::Alt) {
                if self.mode == MotionMode::Examine {
                    MotionAction::Orbit
                } else {
                    MotionAction::LookAround
                }
            } else if modifiers.contains(&Modifier::Shift) {
                log::info!("dolly");
                MotionAction::Dolly
            } else if modifiers.contains(&Modifier::Ctrl) {
                MotionAction::Pan
            } else {
                log::info!("default");
                if self.mode == MotionMode::Examine {
                    MotionAction::Orbit
                } else {
                    MotionAction::LookAround
                }
            }
        } else {
            MotionAction::Orbit
        };
        self.motion(position, action);
    }
}
