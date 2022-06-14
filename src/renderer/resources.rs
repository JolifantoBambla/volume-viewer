use wgpu::{TextureView, Extent3d};

#[readonly::make]
pub struct Texture {
    pub view: TextureView,
    pub extent: Extent3d,
}

impl Texture {
    pub fn is_3d(&self) -> bool {
        self.extent.depth_or_array_layers > 1u32
    }
}
