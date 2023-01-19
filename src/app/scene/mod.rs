use crate::app::scene::camera::Camera;
use crate::resource::VolumeManager;
use crate::Event;
use glam::{Mat4, UVec2};
use wgpu::SubmissionIndex;
use wgpu_framework::event::lifecycle::{OnCommandsSubmitted, Update};
use wgpu_framework::event::window::{OnUserEvent, OnWindowEvent};
use wgpu_framework::geometry::bounds::Bounds3;
use wgpu_framework::input::Input;
use crate::app::scene::volume::VolumeSceneObject;

pub mod camera;
pub mod volume;

pub struct MultiChannelVolumeScene {
    camera: Camera,
    volume: VolumeSceneObject,
}

impl MultiChannelVolumeScene {
    pub fn new(window_size: UVec2, volume_manager: VolumeManager) -> Self {
        // TODO: use framework::camera instead
        // TODO: refactor these params
        let distance_from_center = 500.;
        let camera_speed = 5.0;
        let camera = Camera::new(window_size, distance_from_center, camera_speed);

        let volume = VolumeSceneObject::new_page_table_volume(volume_manager);

        Self {
            camera,
            volume,
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }
    pub fn volume_transform(&self) -> Mat4 {
        self.volume.volume_transform()
    }
    pub fn volume_manager(&self) -> &VolumeManager {
        &self.volume.volume_manager()
    }
    pub fn volume_manager_mut(&mut self) -> &mut VolumeManager {
        self.volume.volume_manager_mut()
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
