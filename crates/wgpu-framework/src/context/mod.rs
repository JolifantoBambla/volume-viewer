use crate::event::window::OnResize;
use std::sync::Arc;
use wgpu;
use wgpu::{Adapter, Device, Instance, Queue, Surface, SurfaceConfiguration, TextureUsages};
use winit::window::Window;

/// Helper struct for constructing a `GPUContext`.
#[derive(Clone, Debug)]
pub struct ContextDescriptor<'a> {
    /// see `wgpu::Instance::new`
    pub backends: wgpu::Backends,

    ///
    pub request_adapter_options: wgpu::RequestAdapterOptions<'a>,

    ///
    pub required_features: wgpu::Features,

    ///
    pub optional_features: wgpu::Features,

    ///
    pub required_limits: wgpu::Limits,

    ///
    pub required_downlevel_capabilities: wgpu::DownlevelCapabilities,
}

impl<'a> Default for ContextDescriptor<'a> {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::all(),
            request_adapter_options: wgpu::RequestAdapterOptions::default(),
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            required_downlevel_capabilities: wgpu::DownlevelCapabilities::default(),
        }
    }
}

#[derive(Debug)]
pub struct Gpu {
    device: Device,
    queue: Queue,
}

impl Gpu {
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

#[derive(Clone, Debug)]
pub enum SurfaceTarget<'a> {
    Window(&'a Window),
    #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
    Canvas(&'a web_sys::HtmlCanvasElement),
    #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
    OffscreenCanvas(&'a web_sys::OffscreenCanvas),
}

impl<'a> SurfaceTarget<'a> {
    pub fn create_surface(&self, instance: &Instance) -> Surface {
        match self {
            SurfaceTarget::Window(w) => unsafe { instance.create_surface(w) },
            #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
            SurfaceTarget::Canvas(c) => instance.create_surface_from_canvas(c),
            #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
            SurfaceTarget::OffscreenCanvas(c) => instance.create_surface_from_offscreen_canvas(c),
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            SurfaceTarget::Window(w) => w.inner_size().width,
            #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
            SurfaceTarget::Canvas(c) => c.width(),
            #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
            SurfaceTarget::OffscreenCanvas(c) => c.width(),
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            SurfaceTarget::Window(w) => w.inner_size().height,
            #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
            SurfaceTarget::Canvas(c) => c.height(),
            #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
            SurfaceTarget::OffscreenCanvas(c) => c.height(),
        }
    }
}

#[derive(Debug)]
pub struct HeadlessContext {
    instance: Instance,
    adapter: Adapter,
    device_context: Arc<Gpu>,
}

impl HeadlessContext {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.device_context
    }
}

#[derive(Debug)]
pub struct SurfaceContext {
    instance: Instance,
    adapter: Adapter,
    gpu: Arc<Gpu>,
    surface: Surface,
    surface_configuration: SurfaceConfiguration,
}

impl SurfaceContext {
    pub fn configure_surface(&self) {
        self.surface
            .configure(self.gpu.device(), &self.surface_configuration);
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    pub fn surface_configuration(&self) -> &SurfaceConfiguration {
        &self.surface_configuration
    }
}

impl OnResize for SurfaceContext {
    fn on_resize(&mut self, width: u32, height: u32) {
        self.surface_configuration.width = width;
        self.surface_configuration.height = height;
        self.configure_surface();
    }
}

#[derive(Debug)]
pub enum WgpuContext {
    Surface(SurfaceContext),
    Headless(HeadlessContext),
}

impl WgpuContext {
    pub async fn new<'a>(
        context_descriptor: &ContextDescriptor<'a>,
        surface_target: Option<SurfaceTarget<'a>>,
    ) -> Self {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::new(context_descriptor.backends);

        let surface = surface_target
            .as_ref()
            .map(|surface_target| surface_target.create_surface(&instance));

        let mut request_adapter_options = context_descriptor.request_adapter_options.clone();
        request_adapter_options.compatible_surface = if let Some(surface) = surface.as_ref() {
            Some(surface)
        } else {
            None
        };

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&request_adapter_options)
            .await
            .expect("No suitable GPU adapters found on the system!");

        // todo: as soon as there is a newer (working) version of wgpu, use the actual available features
        let adapter_features = wgpu::Features::all(); // adapter.features();
        log::warn!("wgpu 0.14.2 does not query available features. We assume that everything we need is available - might break!");
        assert!(
            adapter_features.contains(context_descriptor.required_features),
            "Adapter does not support required features: {:?}",
            context_descriptor.required_features - adapter_features
        );

        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        assert!(
            downlevel_capabilities.shader_model
                >= context_descriptor
                    .required_downlevel_capabilities
                    .shader_model,
            "Adapter does not support the minimum shader model required: {:?}",
            context_descriptor
                .required_downlevel_capabilities
                .shader_model
        );
        assert!(
            downlevel_capabilities
                .flags
                .contains(context_descriptor.required_downlevel_capabilities.flags),
            "Adapter does not support the downlevel capabilities required: {:?}",
            context_descriptor.required_downlevel_capabilities.flags - downlevel_capabilities.flags
        );

        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
        let needed_limits = context_descriptor
            .required_limits
            .clone()
            .using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: (context_descriptor.optional_features & adapter_features)
                        | context_descriptor.required_features,
                    limits: needed_limits,
                },
                // Tracing isn't supported on the Web target
                Option::None,
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        let gpu = Arc::new(Gpu { device, queue });
        if let Some(surface) = surface {
            let surface_target = surface_target.unwrap();
            let format = surface.get_supported_formats(&adapter)[0];
            let present_mode = surface.get_supported_present_modes(&adapter)[0];
            let alpha_mode = surface.get_supported_alpha_modes(&adapter)[0];
            let surface_configuration = SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format,
                width: surface_target.width(),
                height: surface_target.height(),
                present_mode,
                alpha_mode,
            };
            surface.configure(gpu.device(), &surface_configuration);

            Self::Surface(SurfaceContext {
                instance,
                adapter,
                gpu,
                surface,
                surface_configuration,
            })
        } else {
            Self::Headless(HeadlessContext {
                instance,
                adapter,
                device_context: gpu,
            })
        }
    }

    pub fn instance(&self) -> &Instance {
        match self {
            WgpuContext::Surface(s) => s.instance(),
            WgpuContext::Headless(h) => h.instance(),
        }
    }

    pub fn adapter(&self) -> &Adapter {
        match self {
            WgpuContext::Surface(s) => s.adapter(),
            WgpuContext::Headless(h) => h.adapter(),
        }
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        match self {
            WgpuContext::Surface(s) => s.gpu(),
            WgpuContext::Headless(h) => h.gpu(),
        }
    }

    pub fn surface_context(&self) -> &SurfaceContext {
        match self {
            WgpuContext::Surface(s) => s,
            WgpuContext::Headless(_) => {
                panic!("surface_context called on a headless context")
            }
        }
    }

    pub fn surface_context_mut(&mut self) -> &mut SurfaceContext {
        match self {
            WgpuContext::Surface(s) => s,
            WgpuContext::Headless(_) => {
                panic!("surface_context_mut called on a headless context")
            }
        }
    }
}
