use crate::app::scene::camera::Camera;
use crate::resource::VolumeManager;
use crate::Event;
use glam::{Mat4, UVec2};
use wgpu_framework::event::lifecycle::Update;
use wgpu_framework::event::window::{OnUserEvent, OnWindowEvent};
use wgpu_framework::geometry::bounds::Bounds3;
use wgpu_framework::input::Input;

pub mod camera;
pub mod volume;

pub struct MultiChannelVolumeScene {
    camera: Camera,
    volume_transform: Mat4,
    volume_manager: VolumeManager,
}

impl MultiChannelVolumeScene {
    pub fn new(window_size: UVec2, volume_manager: VolumeManager) -> Self {
        // TODO: use framework::camera instead
        // TODO: refactor these params
        let distance_from_center = 500.;
        let camera_speed = 5.0;
        let camera = Camera::new(window_size, distance_from_center, camera_speed);
        // todo: refactor multi-volume into scene object or whatever
        // the volume is a unit cube ([0,1]^3)
        // we translate it s.t. its center is the origin and scale it to its original dimensions
        let volume_transform = glam::Mat4::from_scale(volume_manager.normalized_volume_size())
            .mul_mat4(&glam::Mat4::from_translation(glam::Vec3::new(
                -0.5, -0.5, -0.5,
            )));

        Self {
            camera,
            volume_transform,
            volume_manager,
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
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
    fn update(&mut self, input: &Input) {
        self.camera.update(input);
    }
}

// todo: refactor into camera's update (using input)
impl OnUserEvent for MultiChannelVolumeScene {
    type UserEvent = Event<()>;

    fn on_user_event(&mut self, event: &Self::UserEvent) {
        if let Self::UserEvent::Window(event) = event {
            self.camera.on_window_event(event);
        }
    }
}
