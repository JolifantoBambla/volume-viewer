use wgpu;
use winit;

/// Helper struct for constructing a `GPUContext`.
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

///
pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: Option<wgpu::Surface>,
    pub surface_configuration: Option<wgpu::SurfaceConfiguration>,
}

impl GPUContext {
    async fn create_wgpu_base<'a>(
        context_descriptor: &ContextDescriptor<'a>,
    ) -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::new(context_descriptor.backends);

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&context_descriptor.request_adapter_options)
            .await
            .expect("No suitable GPU adapters found on the system!");

        let adapter_features = adapter.features();
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
        (instance, adapter, device, queue)
    }

    pub async fn new<'a>(context_descriptor: &ContextDescriptor<'a>) -> Self {
        let (instance, adapter, device, queue) = Self::create_wgpu_base(context_descriptor).await;
        Self {
            instance,
            adapter,
            device,
            queue,
            surface: None,
            surface_configuration: None,
        }
    }

    fn choose_surface_format(&self) -> wgpu::TextureFormat {
        if self.surface.is_none() {
            panic!("No surface set!");
        }
        self.surface
            .as_ref()
            .unwrap()
            .get_supported_formats(&self.adapter)[0]
    }

    fn choose_present_mode(&self) -> wgpu::PresentMode {
        if self.surface.is_none() {
            panic!("No surface set!");
        }
        self.surface
            .as_ref()
            .unwrap()
            .get_supported_present_modes(&self.adapter)[0]
    }

    fn choose_alpha_mode(&self) -> wgpu::CompositeAlphaMode {
        if self.surface.is_none() {
            panic!("No surface set!");
        }
        self.surface
            .as_ref()
            .unwrap()
            .get_supported_alpha_modes(&self.adapter)[0]
    }

    fn configure_surface(&self) {
        if self.surface.is_none() || self.surface_configuration.is_none() {
            panic!("No surface or surface configuration set!");
        }
        self.surface
            .as_ref()
            .unwrap()
            .configure(&self.device, self.surface_configuration.as_ref().unwrap());
    }

    pub fn with_surface_from_window(mut self, window: &winit::window::Window) -> Self {
        self.surface = unsafe { Some(self.instance.create_surface(&window)) };

        self.surface_configuration = Some(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.choose_surface_format(),
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: self.choose_present_mode(),
            alpha_mode: self.choose_alpha_mode(),
        });

        self.configure_surface();

        self
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
    pub fn with_surface_from_offscreen_canvas(mut self, canvas: &web_sys::OffscreenCanvas) -> Self {
        self.surface = Some(self.instance.create_surface_from_offscreen_canvas(canvas));

        self.surface_configuration = Some(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.choose_surface_format(),
            width: canvas.width(),
            height: canvas.height(),
            present_mode: self.choose_present_mode(),
            alpha_mode: self.choose_alpha_mode(),
        });

        self.configure_surface();

        self
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "emscripten")))]
    pub fn with_surface_from_canvas(mut self, canvas: &web_sys::HtmlCanvasElement) -> Self {
        self.surface = Some(self.instance.create_surface_from_canvas(canvas));

        self.surface_configuration = Some(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.choose_surface_format(),
            width: canvas.width(),
            height: canvas.height(),
            present_mode: self.choose_present_mode(),
            alpha_mode: self.choose_alpha_mode(),
        });

        self.configure_surface();

        self
    }

    pub fn resize_surface(&mut self, width: u32, height: u32) {
        if self.surface.is_none() || self.surface_configuration.is_none() {
            panic!("No surface or surface configuration set!");
        }
        let mut surface_configuration = self.surface_configuration.as_mut().unwrap();
        surface_configuration.width = width;
        surface_configuration.height = height;
        self.configure_surface();
    }
}
