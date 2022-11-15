use glam::Vec4;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CreateOptions {
    #[serde(rename = "maxVisibleChannels")]
    pub max_visible_channels: u32,

    #[serde(rename = "maxResolutions")]
    pub max_resolutions: u32,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum RenderMode {
    /// Renders a bricked volume by stepping from brick to brick in a voxel line, and sampling the
    /// fetched bricks.
    #[serde(rename = "grid_traversal")]
    GridTraversal,

    /// Renders a bricked volume by sampling a ray at regular intervals and fetching the
    /// corresponding brick for each sample.
    #[serde(rename = "direct")]
    Direct,
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
}

impl From<Color> for Vec4 {
    fn from(color: Color) -> Self {
        Vec4::new(color.r, color.g, color.b, 1.)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChannelSettings {
    /// This channel's name.
    #[serde(rename = "channelName")]
    pub channel_name: String,

    /// This channel's index in the multi-channel volume.
    #[serde(rename = "channelIndex")]
    pub channel_index: u32,

    /// The highest level of detail to use for this channel.
    /// Levels of detail are ordered from highest (0) to lowest (n).
    /// If this `max_lod` > n, defaults to n.
    #[serde(rename = "maxLoD")]
    pub max_lod: u32,

    /// The lowest level of detail to use for this channel.
    /// Levels of detail are ordered from highest (0) to lowest (n).
    /// If this `min_lod` > n, defaults to n.
    #[serde(rename = "minLoD")]
    pub min_lod: u32,

    /// The lower threshold for this channel.
    /// Every value below it is treated as 0 during rendering.
    /// Range: [0,1]
    #[serde(rename = "thresholdLower")]
    pub threshold_lower: f32,

    /// The upper threshold for this channel.
    /// Every value above it is treated as 0 during rendering.
    /// Range: [0,1]
    #[serde(rename = "thresholdUpper")]
    pub threshold_upper: f32,

    /// The color to use for rendering this channel.
    /// All components are in range [0,1]
    pub color: Color,

    /// If true, this channel gets rendered, otherwise it does not get rendered.
    pub visible: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultiChannelVolumeRendererSettings {
    #[serde(rename = "createOptions")]
    pub create_options: CreateOptions,

    #[serde(rename = "renderMode")]
    pub render_mode: RenderMode,

    /// The `step_scale` is used to scale steps along a ray, where a scale of 1 is sampling roughly
    /// corresponds to one sample per voxel, and a smaller scale corresponds to smaller steps.
    /// Range: [0,n]
    #[serde(rename = "stepScale")]
    pub step_scale: f32,

    /// The maximum number of samples to take for each ray.
    #[serde(rename = "maxSteps")]
    pub max_steps: u32,

    /// The background color to use for rendering.
    #[serde(rename = "backgroundColor")]
    pub background_color: Color,

    #[serde(rename = "channelSettings")]
    pub channel_settings: Vec<ChannelSettings>,
}

impl MultiChannelVolumeRendererSettings {
    pub fn get_visible_channel_indices(&self) -> Vec<u32> {
        self.channel_settings
            .iter()
            .filter(|c| c.visible)
            .map(|c| c.channel_index)
            .collect()
    }

    pub fn get_sorted_visible_channel_indices(&self) -> Vec<u32> {
        // todo: sort by channel importance
        self.get_visible_channel_indices()
    }
}
