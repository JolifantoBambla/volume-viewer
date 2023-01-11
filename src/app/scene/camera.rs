use wgpu_framework::event::lifecycle::Update;
use wgpu_framework::event::window::OnResize;
use wgpu_framework::input::mouse::MouseEvent;
use wgpu_framework::input::{Event, Input};
use wgpu_framework::scene::camera::{Camera, CameraView, OrthographicProjection, PerspectiveProjection, Projection};
use wgpu_framework::scene::transform::util::Orbit;
use wgpu_framework::scene::transform::{Transform, Transformable};
use glam::{Mat4, Vec2, Vec3};
use winit::event::VirtualKeyCode;
use wgpu_framework::geometry::bounds::Bounds3;

#[derive(Copy, Clone, Debug)]
pub struct OrbitCamera {
    camera: Camera,
    speed: f32,
    zoom_speed: f32,
    orthographic: OrthographicProjection,
    perspective: PerspectiveProjection,
}

impl OrbitCamera {
    pub fn new(view: CameraView, window_size: Vec2, near: f32, far: f32, speed: f32, zoom_speed: f32) -> Self {
        let orthographic = OrthographicProjection::new(Bounds3::new(
            (window_size * - 0.5).extend(near),
            (window_size * 0.5).extend(far),
        ));
        let perspective = PerspectiveProjection::new(
            f32::to_radians(45.0),
            window_size.x / window_size.y,
            near,
            far,
        );
        let camera = Camera::new(view, Projection::Perspective(perspective));
        Self {
            camera,
            speed,
            zoom_speed,
            orthographic,
            perspective,
        }
    }
    pub fn view(&self) -> Mat4 {
        self.camera.view_mat()
    }
    pub fn projection(&self) -> Mat4 {
        self.camera.projection_mat()
    }
    pub fn is_orthographic(&self) -> bool {
        match self.camera.projection() {
            Projection::Orthographic(_) => true,
            _ => false
        }
    }
}

impl Transformable for OrbitCamera {
    fn transform(&self) -> &Transform {
        self.camera.transform()
    }

    fn transform_mut(&mut self) -> &mut Transform {
        self.camera.transform_mut()
    }
}

impl Orbit for OrbitCamera {
    fn target(&self) -> Vec3 {
        self.camera.view().center_of_projection()
    }

    fn set_target(&mut self, target: Vec3) {
        self.camera.view_mut().set_center_of_projection(target);
    }
}

impl OnResize for OrbitCamera {
    fn on_resize(&mut self, width: u32, height: u32) {
        self.camera.projection_mut().on_resize(width, height)
    }
}

impl Update for OrbitCamera {
    fn update(&mut self, input: &Input) {
        for e in input.events() {
            match e {
                Event::Mouse(m) => match m {
                    MouseEvent::Move(m) => {
                        if m.state().left_button_pressed() {
                            self.orbit(m.delta(), false);
                        }
                        if m.state().right_button_pressed() {
                            let translation = m.delta() * self.speed * 20.;
                            let distance_to_target = self.distance_to_target();
                            self.camera.move_right(translation.x);
                            self.camera.move_down(translation.y);
                            self.set_target(
                                self.camera.transform().position()
                                    + self.camera.transform().forward() * distance_to_target,
                            );
                        }
                    }
                    MouseEvent::Scroll(s) => {
                        self.camera.zoom_in(
                            s.delta().abs().min(1.) * s.delta().signum() * self.zoom_speed,
                        );
                    }
                    _ => {}
                },
                Event::Keyboard(k) => {
                    if k.is_pressed() {
                        match k.key() {
                            VirtualKeyCode::D => self.camera.move_right(self.speed),
                            VirtualKeyCode::A => self.camera.move_left(self.speed),
                            VirtualKeyCode::W | VirtualKeyCode::Up => self.camera.move_forward(self.speed),
                            VirtualKeyCode::S | VirtualKeyCode::Down => self.camera.move_backward(self.speed),
                            VirtualKeyCode::C => {
                                match self.camera.projection() {
                                    Projection::Perspective(_) => {
                                        self.camera.set_projection(Projection::Orthographic(self.orthographic));
                                    },
                                    Projection::Orthographic(_) => {
                                        self.camera.set_projection(Projection::Perspective(self.perspective));
                                    }
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}