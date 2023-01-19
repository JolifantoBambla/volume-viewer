use glam::{Mat3, Mat4, UVec2, Vec2, Vec3};
use wgpu_framework::event::lifecycle::Update;
use wgpu_framework::event::window::OnWindowEvent;
use wgpu_framework::geometry::bounds::{Bounds, Bounds2, Bounds3};
use wgpu_framework::input::Input;
use winit::event::{
    ElementState, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};

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

    orthographic: Projection,
    perspective: Projection,
    resolution: Vec2,
    last_mouse_position: Vec2,
    left_mouse_pressed: bool,
    right_mouse_pressed: bool,

    speed: f32,
}

impl Camera {
    pub fn new(window_size: UVec2, distance_from_center: f32, speed: f32) -> Self {
        let near = 0.0001;
        let far = 1000.0;

        let resolution = Vec2::new(window_size.x as f32, window_size.y as f32);
        let perspective = Projection::new_perspective(
            f32::to_radians(45.),
            window_size.x as f32 / window_size.y as f32,
            near,
            far,
        );
        let orthographic = Projection::new_orthographic(Bounds3::new(
            (resolution * -0.5).extend(near),
            (resolution * 0.5).extend(far),
        ));

        let view = CameraView::new(
            Vec3::new(1., 1., 1.) * distance_from_center,
            Vec3::new(0., 0., 0.),
            Vec3::new(0., 1., 0.),
        );

        // todo: use input instead
        let last_mouse_position = Vec2::new(0., 0.);
        let left_mouse_pressed = false;
        let right_mouse_pressed = false;

        Self {
            view,
            projection: perspective,

            orthographic,
            perspective,
            resolution,
            last_mouse_position,
            left_mouse_pressed,
            right_mouse_pressed,

            speed,
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
}

impl Update for Camera {
    fn update(&mut self, _input: &Input) {
        // todo: update camera here instead of OnWindowEvent
    }
}

impl OnWindowEvent for Camera {
    fn on_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(virtual_keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match virtual_keycode {
                VirtualKeyCode::D => self.view.move_right(self.speed),
                VirtualKeyCode::A => self.view.move_left(self.speed),
                VirtualKeyCode::W | VirtualKeyCode::Up => self.view.move_forward(self.speed),
                VirtualKeyCode::S | VirtualKeyCode::Down => self.view.move_backward(self.speed),
                VirtualKeyCode::C => {
                    if self.projection().is_orthographic() {
                        self.set_projection(self.perspective);
                    } else {
                        self.set_projection(self.orthographic);
                    }
                }
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let mouse_position = Vec2::new(position.x as f32, position.y as f32);
                let delta = (mouse_position - self.last_mouse_position) / self.resolution;
                self.last_mouse_position = mouse_position;

                if self.left_mouse_pressed {
                    self.view.orbit(delta, false);
                } else if self.right_mouse_pressed {
                    let translation = delta * self.speed * 20.;
                    self.view.move_right(translation.x);
                    self.view.move_down(translation.y);
                }
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::PixelDelta(delta),
                ..
            } => {
                self.view.move_forward(
                    (f64::min(delta.y.abs(), 1.) * delta.y.signum()) as f32 * self.speed,
                );
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => {
                    self.left_mouse_pressed = *state == ElementState::Pressed;
                }
                MouseButton::Right => {
                    self.right_mouse_pressed = *state == ElementState::Pressed;
                }
                _ => {}
            },
            _ => {}
        }
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
