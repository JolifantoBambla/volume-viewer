use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use glam::{Mat3, Mat4, Vec2, Vec3};

pub trait Camera {}

pub struct OrthographicCamera {}

impl Camera for OrthographicCamera {}

pub struct PerspectiveCamera {}

impl Camera for PerspectiveCamera {}

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

fn is_close_to_zero(val: f32) -> bool {
    val.abs() < f32::EPSILON
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
                self.orbit(delta, self.mode == MotionMode::Trackball);
            }
            MotionAction::Dolly => {
                self.dolly(delta);
            }
            MotionAction::Pan => {
                self.pan(delta);
            }
            MotionAction::LookAround => {
                if self.mode == MotionMode::Trackball {
                    self.trackball(position);
                } else {
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
