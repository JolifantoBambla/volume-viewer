use serde::{Deserialize, Serialize};
use winit::event::WindowEvent;

pub mod conversion;
pub mod handler;

use crate::renderer::settings::{Color, RenderMode};

pub struct RawArrayReceived {
    pub data: Vec<u16>,
    pub shape: Vec<u32>,
}

#[derive(Deserialize, Serialize)]
pub struct Range<T> {
    pub min: T,
    pub max: T,
}

#[derive(Deserialize, Serialize)]
pub enum ChannelSettingsChange {
    #[serde(rename = "lod")]
    LoD(Range<u32>),

    #[serde(rename = "threshold")]
    Threshold(Range<f32>),

    #[serde(rename = "color")]
    Color(Color),

    #[serde(rename = "visible")]
    Visible(bool),
}

#[derive(Deserialize, Serialize)]
pub struct ChannelSetting {
    #[serde(rename = "channelIndex")]
    pub channel_index: u32,

    #[serde(rename = "channelSetting")]
    pub channel_setting: ChannelSettingsChange,
}

#[derive(Deserialize, Serialize)]
pub enum SettingsChange {
    #[serde(rename = "renderMode")]
    RenderMode(RenderMode),

    #[serde(rename = "stepScale")]
    StepScale(f32),

    #[serde(rename = "maxSteps")]
    MaxSteps(u32),

    #[serde(rename = "backgroundColor")]
    BackgroundColor(Color),

    #[serde(rename = "channelSetting")]
    ChannelSetting(ChannelSetting),
}

pub enum Event<'a, T: 'static> {
    Window(WindowEvent<'a>),
    RawArray(RawArrayReceived),
    Settings(SettingsChange),
    Custom(T),
}

