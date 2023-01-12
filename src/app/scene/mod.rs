use glam::{Mat4, UVec2, Vec2, Vec3};
use winit::event::{ElementState, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent};
use wgpu_framework::event::lifecycle::Update;
use wgpu_framework::event::window::OnUserEvent;
use wgpu_framework::input::Input;
use crate::Event;
use crate::renderer::camera::{Camera, CameraView, Projection};
use crate::renderer::geometry::Bounds3D;
use crate::resource::VolumeManager;

// todo: refactor into camera's fields
const TRANSLATION_SPEED: f32 = 5.0;
const NEAR: f32 = 0.0001;
const FAR: f32 = 1000.0;

pub struct MultiChannelVolumeScene {
    camera: Camera,
    volume_transform: Mat4,
    volume_manager: VolumeManager,

    // todo: refactor into camera
    orthographic: Projection,
    perspective: Projection,
    resolution: Vec2,
    last_mouse_position: Vec2,
    left_mouse_pressed: bool,
    right_mouse_pressed: bool,
}

impl MultiChannelVolumeScene {
    pub fn new(
        window_size: UVec2,
        volume_manager: VolumeManager,
    ) -> Self {
        // TODO: use framework::camera instead
        // TODO: refactor these params
        let distance_from_center = 500.;
        let resolution = Vec2::new(window_size.x as f32, window_size.y as f32);
        let perspective = Projection::new_perspective(
            f32::to_radians(45.),
            window_size.x as f32 / window_size.y as f32,
            NEAR,
            FAR,
        );
        let orthographic = Projection::new_orthographic(Bounds3D::new(
            (resolution * -0.5).extend(NEAR),
            (resolution * 0.5).extend(FAR),
        ));
        let camera = Camera::new(
            CameraView::new(
                Vec3::new(1., 1., 1.) * distance_from_center,
                Vec3::new(0., 0., 0.),
                Vec3::new(0., 1., 0.),
            ),
            perspective,
        );
        // todo: refactor multi-volume into scene object or whatever
        // the volume is a unit cube ([0,1]^3)
        // we translate it s.t. its center is the origin and scale it to its original dimensions
        let volume_transform = glam::Mat4::from_scale(volume_manager.normalized_volume_size())
            .mul_mat4(&glam::Mat4::from_translation(glam::Vec3::new(
                -0.5, -0.5, -0.5,
            )));

        let last_mouse_position = Vec2::new(0., 0.);
        let left_mouse_pressed = false;
        let right_mouse_pressed = false;

        Self {
            camera,
            volume_transform,
            volume_manager,

            orthographic,
            perspective,
            resolution,
            last_mouse_position,
            left_mouse_pressed,
            right_mouse_pressed,
        }
    }

    pub fn camera(&self) -> Camera {
        self.camera
    }
    pub fn volume_transform(&self) -> Mat4 {
        self.volume_transform
    }
    pub fn volume_manager(&self) -> &VolumeManager {
        &self.volume_manager
    }
    pub fn volume_manager_mut(&mut self) -> &mut VolumeManager {
        &mut self.volume_manager
    }
}

impl Update for MultiChannelVolumeScene {
    fn update(&mut self, _input: &Input) {
        // todo: update camera
    }
}

// todo: refactor into camera's update (using input)
impl OnUserEvent for MultiChannelVolumeScene {
    type UserEvent = Event<()>;

    fn on_user_event(&mut self, event: &Self::UserEvent) {
        match event {
            // todo: use input & update instead
            Self::UserEvent::Window(event) => match event {
                WindowEvent::KeyboardInput {
                    input:
                    KeyboardInput {
                        virtual_keycode: Some(virtual_keycode),
                        state: ElementState::Pressed,
                        ..
                    },
                    ..
                } => match virtual_keycode {
                    VirtualKeyCode::D => self.camera.view.move_right(TRANSLATION_SPEED),
                    VirtualKeyCode::A => self.camera.view.move_left(TRANSLATION_SPEED),
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.camera.view.move_forward(TRANSLATION_SPEED)
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.camera.view.move_backward(TRANSLATION_SPEED)
                    }
                    VirtualKeyCode::C => {
                        if self.camera.projection().is_orthographic() {
                            self.camera.set_projection(self.perspective);
                        } else {
                            self.camera.set_projection(self.orthographic);
                        }
                    }
                    _ => {}
                },
                WindowEvent::CursorMoved { position, .. } => {
                    let mouse_position = Vec2::new(position.x as f32, position.y as f32);
                    let delta = (mouse_position - self.last_mouse_position) / self.resolution;
                    self.last_mouse_position = mouse_position;

                    if self.left_mouse_pressed {
                        self.camera.view.orbit(delta, false);
                    } else if self.right_mouse_pressed {
                        let translation = delta * TRANSLATION_SPEED * 20.;
                        self.camera.view.move_right(translation.x);
                        self.camera.view.move_down(translation.y);
                    }
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::PixelDelta(delta),
                    ..
                } => {
                    self.camera.view.move_forward(
                        (f64::min(delta.y.abs(), 1.) * delta.y.signum()) as f32 * TRANSLATION_SPEED,
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
            },
            _ => {}
        }
    }
}
