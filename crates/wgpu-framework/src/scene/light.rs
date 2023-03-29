use glam::Vec3;
use serde::Deserialize;

/// The light emitted by a `LightSource`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Deserialize, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Light {
    /// The light's color.
    color: Vec3,

    /// The light's intensity.
    intensity: f32,
}

impl Light {
    pub fn new(color: Vec3, intensity: f32) -> Self {
        Self { color, intensity }
    }
    pub fn color(&self) -> Vec3 {
        self.color
    }
    pub fn intensity(&self) -> f32 {
        self.intensity
    }
    pub fn set_color(&mut self, color: Vec3) {
        self.color = color;
    }
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
    }
}

impl Default for Light {
    fn default() -> Self {
        Self {
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

/// A directional light source.
#[derive(Copy, Clone, Debug, Deserialize)]
pub struct DirectionalLight {
    /// The direction in which light is emitted.
    direction: Vec3,
}

impl DirectionalLight {
    /// Constructs a new `DirectionalLight` emitting light in the given direction.
    /// The given direction gets normalized.
    pub fn new(direction: Vec3) -> Self {
        Self {
            direction: direction.normalize(),
        }
    }
    pub fn direction(&self) -> Vec3 {
        self.direction
    }
}

#[derive(Copy, Clone, Debug, Deserialize)]
pub struct PointLight {
    position: Vec3,
}

impl PointLight {
    pub fn new(position: Vec3) -> Self {
        Self { position }
    }
    pub fn position(&self) -> Vec3 {
        self.position
    }
}

#[derive(Copy, Clone, Debug, Deserialize)]
pub enum LightSourceType {
    Ambient,
    #[serde(rename = "directional")]
    Directional(DirectionalLight),
    #[serde(rename = "point")]
    Point(PointLight),
}

#[derive(Copy, Clone, Debug, Deserialize)]
pub struct LightSource {
    light: Light,
    source: LightSourceType,
}

impl LightSource {
    pub fn new(light: Light, source: LightSourceType) -> Self {
        Self { light, source }
    }

    pub fn new_ambient(color: Vec3) -> Self {
        Self::new(
            Light {
                color,
                ..Default::default()
            },
            LightSourceType::Ambient,
        )
    }

    pub fn new_directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self::new(
            Light::new(color, intensity),
            LightSourceType::Directional(DirectionalLight::new(direction)),
        )
    }
    pub fn new_point(position: Vec3, color: Vec3, intensity: f32) -> Self {
        Self::new(
            Light::new(color, intensity),
            LightSourceType::Point(PointLight::new(position)),
        )
    }
    pub fn light(&self) -> Light {
        self.light
    }
    pub fn source(&self) -> LightSourceType {
        self.source
    }
}
