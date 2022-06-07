use wgpu;
use winit;

use wasm_bindgen::prelude::*;

// todo: proper documentation

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

    // todo: go for canvas instead? would break compatibility with native platforms - do I want to support them?
    ///
    pub window: Option<winit::window::Window>,
}

impl<'a> Default for ContextDescriptor<'a> {
    fn default() -> Self {
        Self{
            backends: wgpu::Backends::all(),
            request_adapter_options: wgpu::RequestAdapterOptions::default(),
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            required_downlevel_capabilities: wgpu::DownlevelCapabilities::default(),
            window: None,
        }
    }
}

/// Holds all access points to
pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: Option<wgpu::Surface>,
}

impl GPUContext {
    pub async fn new<'a>(context_descriptor: &ContextDescriptor<'a>) -> Self {
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

        let downlevel_capabilities = adapter.get_downlevel_properties();
        assert!(
            downlevel_capabilities.shader_model >= context_descriptor.required_downlevel_capabilities.shader_model,
            "Adapter does not support the minimum shader model required: {:?}",
            context_descriptor.required_downlevel_capabilities.shader_model
        );
        assert!(
            downlevel_capabilities
                .flags
                .contains(context_descriptor.required_downlevel_capabilities.flags),
            "Adapter does not support the downlevel capabilities required: {:?}",
            context_descriptor.required_downlevel_capabilities.flags - downlevel_capabilities.flags
        );

        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
        let needed_limits = context_descriptor.required_limits.clone().using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: (context_descriptor.optional_features & adapter_features) | context_descriptor.required_features,
                    limits: needed_limits,
                },
                // Tracing isn't supported on the Web target
                Option::None
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        let surface: Option<wgpu::Surface> = if context_descriptor.window.is_some() {
            unsafe {
                Some(instance.create_surface(&context_descriptor.window.as_ref().unwrap()))
            }
        } else {
            None
        };

        Self {
            instance,
            adapter,
            device,
            queue,
            surface,
        }
    }
}
