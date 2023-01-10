#[cfg(target_arch = "wasm32")]
use web_sys::{HtmlCanvasElement, OffscreenCanvas};
use winit::dpi::PhysicalSize;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
pub struct AttachToParentConfig {
    parent_id: Option<String>,
    size: PhysicalSize<u32>,
}

impl AttachToParentConfig {
    pub fn parent_id(&self) -> &Option<String> {
        &self.parent_id
    }
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
pub enum CanvasConfig {
    Canvas(HtmlCanvasElement),
    OffscreenCanvas(OffscreenCanvas),
    CanvasId(String),
    AttachToParent(AttachToParentConfig),
}

#[derive(Clone, Debug)]
pub struct WindowConfig {
    title: String,
    #[cfg(not(target_arch = "wasm32"))]
    size: PhysicalSize<u32>,
    #[cfg(target_arch = "wasm32")]
    canvas_config: CanvasConfig,
}

impl WindowConfig {
    pub fn new(title: String, size: PhysicalSize<u32>) -> Self {
        Self {
            title,
            #[cfg(not(target_arch = "wasm32"))]
            size,
            #[cfg(target_arch = "wasm32")]
            canvas_config: CanvasConfig::AttachToParent(AttachToParentConfig {parent_id: None, size}),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_with_canvas(title: String, canvas: HtmlCanvasElement) -> Self {
        Self {
            title,
            canvas_config: CanvasConfig::Canvas(canvas),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_with_offscreen_canvas(title: String, offscreen_canvas: OffscreenCanvas) -> Self {
        Self {
            title,
            canvas_config: CanvasConfig::OffscreenCanvas(offscreen_canvas),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_with_canvas_id(title: String, canvas_id: String) -> Self {
        Self {
            title,
            canvas_config: CanvasConfig::CanvasId(canvas_id),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_with_parent_id(title: String, size: PhysicalSize<u32>, parent_id: String) -> Self {
        Self {
            title,
            canvas_config: CanvasConfig::AttachToParent(AttachToParentConfig {parent_id: Some(parent_id), size}),
        }
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    #[cfg(target_arch = "wasm32")]
    pub fn canvas_config(&self) -> &CanvasConfig {
        &self.canvas_config
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            size: PhysicalSize {
                width: 800,
                height: 600,
            },
            #[cfg(target_arch = "wasm32")]
            canvas_config: CanvasConfig::AttachToParent(AttachToParentConfig {
                parent_id: None,
                size: PhysicalSize {
                    width: 800,
                    height: 600,
                }
            }),
        }
    }
}
