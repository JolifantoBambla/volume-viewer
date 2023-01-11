use glam::{Mat4, UVec2, Vec2, Vec3};
use crate::renderer::camera::{Camera, CameraView, Projection};
use crate::renderer::geometry::Bounds3D;
use crate::resource::VolumeManager;

pub struct MultiChannelVolumeScene {
    camera: Camera,
    volume_transform: Mat4,
    volume_manager: VolumeManager,
}

impl MultiChannelVolumeScene {
    pub fn new(
        window_size: UVec2,
        volume_manager: VolumeManager,
    ) -> Self {
        // todo: refactor into fields
        const TRANSLATION_SPEED: f32 = 5.0;
        const NEAR: f32 = 0.0001;
        const FAR: f32 = 1000.0;

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
        Self {
            camera,
            volume_transform,
            volume_manager,
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
}