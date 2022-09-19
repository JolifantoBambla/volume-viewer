use serde::{Deserialize, Serialize};
use winit::event::WindowEvent;

pub mod conversion;
pub mod handler;

use crate::renderer::settings::RenderMode;

pub struct RawArrayReceived {
    pub data: Vec<u16>,
    pub shape: Vec<u32>,
}

#[derive(Deserialize, Serialize)]
pub enum SettingsChange {
    #[serde(rename = "renderMode")]
    RenderMode(RenderMode),

    #[serde(rename = "stepScale")]
    StepScale(f32),

    #[serde(rename = "threshold")]
    Threshold(f32),

    #[serde(rename = "color")]
    Color(String)
}

pub enum Event<'a, T: 'static> {
    Window(WindowEvent<'a>),
    RawArray(RawArrayReceived),
    Settings(SettingsChange),
    Custom(T),
}

