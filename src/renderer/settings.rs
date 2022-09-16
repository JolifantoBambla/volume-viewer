use serde::{Deserialize, Serialize};

#[repr(u32)]
#[derive(Deserialize, Serialize)]
pub enum RenderMode {
    #[serde(rename = "grid_traversal")]
    GridTraversal,

    #[serde(rename = "direct")]
    Direct,
}

#[derive(Deserialize, Serialize)]
pub struct ChannelSettings {

}

#[derive(Deserialize, Serialize)]
pub struct MultiChannelVolumeRendererSettings {
    #[serde(rename = "renderMode")]
    pub render_mode: RenderMode,

    #[serde(rename = "stepSize")]
    pub step_size: f32,

    #[serde(rename = "maxSteps")]
    pub max_steps: u32,

    #[serde(rename = "thresholdLower")]
    pub threshold_lower: f32,

    #[serde(rename = "thresholdUpper")]
    pub threshold_upper: f32,

    #[serde(rename = "channelSettings")]
    pub channel_settings: Vec<ChannelSettings>,
}