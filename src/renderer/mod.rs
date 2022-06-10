use wgpu;
use wgpu::util::DeviceExt;
use winit;

pub mod full_screen_pass;

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
        Self{
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
    pub async fn new<'a>(context_descriptor: &ContextDescriptor<'a>, window: Option<&winit::window::Window>) -> Self {
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

        let surface: Option<wgpu::Surface> = if window.is_some() {
            unsafe {
                Some(instance.create_surface(&window.as_ref().unwrap()))
            }
        } else {
            None
        };

        let surface_configuration = if surface.is_some() {
            let surface_configuration = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface.as_ref().unwrap().get_preferred_format(&adapter).unwrap(),
                width: window.as_ref().unwrap().inner_size().width,
                height: window.as_ref().unwrap().inner_size().height,
                present_mode: wgpu::PresentMode::Mailbox,
            };
            surface.as_ref().unwrap().configure(&device, &surface_configuration);
            Some(surface_configuration)
        } else {
            None
        };

        Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            surface_configuration,
        }
    }
}

// todo: remove
// this is just a small module where I test stuff
pub mod playground {
    use std::{borrow::Cow, str::FromStr};
    use std::convert::TryInto;
    use bytemuck;
    use winit::{
        event::{self, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::Window
    };
    use wgpu::util::DeviceExt;
    use crate::renderer::{GPUContext, ContextDescriptor};

    pub struct FullScreenPass {
        bind_group_layout: wgpu::BindGroupLayout,
        bind_group: wgpu::BindGroup,
        pipeline: wgpu::RenderPipeline,
    }

    impl FullScreenPass {
        pub fn new(texture_view: &wgpu::TextureView, sampler: &wgpu::Sampler, ctx: &GPUContext) -> Self {
            let shader_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("full_screen_quad.wgsl"))),
            });
            let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None, //Some(&full_screen_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vert_main",
                    buffers: &[]
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "frag_main",
                    targets: &[ctx.surface_configuration.as_ref().unwrap().format.into()],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None
            });
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    }
                ]
            });
            Self {
                bind_group_layout,
                bind_group,
                pipeline,
            }
        }

        pub fn update_bind_group(&mut self, texture_view: &wgpu::TextureView, sampler: &wgpu::Sampler, ctx: &GPUContext) {
            self.bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    }
                ]
            });
        }

        pub fn add_to_command(&self, mut command_encoder: wgpu::CommandEncoder, view: &wgpu::TextureView) -> wgpu::CommandEncoder {
            {
                let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0
                            }),
                            store: true
                        }
                    }],
                    depth_stencil_attachment: None
                });
                rpass.set_pipeline(&self.pipeline);
                rpass.set_bind_group(0, &self.bind_group, &[]);
                rpass.draw(0..6, 0..1);
            }
            command_encoder
        }
    }

    pub struct EdgeDetectPass {
        bind_group_layout: wgpu::BindGroupLayout,
        bind_group: wgpu::BindGroup,
        pipeline: wgpu::ComputePipeline,
        input_texture_width: u32,
        input_texture_height: u32,
    }

    impl EdgeDetectPass {
        pub fn new(input_texture_view: &wgpu::TextureView, output_texture_view: &wgpu::TextureView, input_texture_width: u32, input_texture_height: u32, ctx: &GPUContext) -> Self {
            let shader_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("edgedetect.wgsl"))),
            });
            let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            });
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_texture_view),
                    }
                ],
            });
            Self {
                bind_group_layout,
                bind_group,
                pipeline,
                input_texture_width,
                input_texture_height
            }
        }

        pub fn update_bind_group(&mut self, input_texture_view: &wgpu::TextureView, output_texture_view: &wgpu::TextureView, input_texture_width: u32, input_texture_height: u32, ctx: &GPUContext) {
            self.bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_texture_view),
                    }
                ],
            });
            self.input_texture_width = input_texture_width;
            self.input_texture_height = input_texture_height;
        }

        pub fn add_to_command(&self, mut command_encoder: wgpu::CommandEncoder) -> wgpu::CommandEncoder {
            {
                let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.insert_debug_marker("compute collatz iterations");
                cpass.dispatch(self.input_texture_width / 16, self.input_texture_height / 16, 1);
            }
            command_encoder
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct ZSliceUniforms {
        zslice: i32,
        zmax: f32,
    }

    pub struct ZSlicePass {
        bind_group_layout: wgpu::BindGroupLayout,
        bind_group: wgpu::BindGroup,
        pipeline: wgpu::ComputePipeline,
        input_texture_width: u32,
        input_texture_height: u32,
    }

    impl ZSlicePass {
        pub fn new(input_texture_view: &wgpu::TextureView, output_texture_view: &wgpu::TextureView, input_texture_width: u32, input_texture_height: u32, z_slice_buffer: &wgpu::Buffer, ctx: &GPUContext) -> Self {
            let shader_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("z_slice.wgsl"))),
            });
            let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            });
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_texture_view),
                    },
                    wgpu::BindGroupEntry{
                        binding: 2,
                        resource: z_slice_buffer.as_entire_binding(),
                    }
                ],
            });
            Self {
                bind_group_layout,
                bind_group,
                pipeline,
                input_texture_width,
                input_texture_height
            }
        }

        pub fn update_bind_group(&mut self, input_texture_view: &wgpu::TextureView, output_texture_view: &wgpu::TextureView, input_texture_width: u32, input_texture_height: u32, ctx: &GPUContext) {
            self.bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_texture_view),
                    }
                ],
            });
            self.input_texture_width = input_texture_width;
            self.input_texture_height = input_texture_height;
        }

        pub fn add_to_command(&self, mut command_encoder: wgpu::CommandEncoder) -> wgpu::CommandEncoder {
            {
                let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.insert_debug_marker("compute collatz iterations");
                cpass.dispatch(self.input_texture_width / 16, self.input_texture_height / 16, 1);
            }
            command_encoder
        }
    }

    fn create_storage_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture_extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, texture_view)
    }

    // let's see if this works - data is u16, but we'll get the u8 array buffer instead
    pub fn create_volume_texture(texels: &[u32], extent: wgpu::Extent3d, ctx: &GPUContext) -> Texture {
        let texture = ctx.device.create_texture_with_data(
            &ctx.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            },
            bytemuck::cast_slice(texels),//bytemuck::cast_slice::<u32,u8>(bytemuck::cast_slice::<u16, u32>(texels))
        );
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Texture {
            texture,
            view: texture_view
        }
    }

    pub struct RawVolume {
        pub data: Vec<u32>,
        pub shape: Vec<u32>,
    }

    impl RawVolume {
        pub fn max_per_slice(&self) -> Vec<u32> {
            let stride = (self.shape[1] * self.shape[2]) as usize;
            let mut result: Vec<u32> = Vec::with_capacity(self.shape[0] as _);
            for i in 0..result.capacity() {
                result.push(*self.data[i*stride..(i+1)*stride].iter().max().unwrap());
            }
            result
        }
    }

    pub struct Texture {
        pub texture: wgpu::Texture,
        pub view: wgpu::TextureView,
    }

    pub struct ZSlicer {
        window: winit::window::Window,
        ctx: GPUContext,

        volume_texture: Texture,
        storage_texture: Texture,
        sampler: wgpu::Sampler,

        z_slice_pass: ZSlicePass,
        full_screen_pass: FullScreenPass,

        z_slider: web_sys::HtmlInputElement,
        z_slice_uniform_buffer: wgpu::Buffer,
        z_max: Vec<u32>,
    }

    impl ZSlicer {
        pub async fn new(window: Window, volume: RawVolume, z_slider_id: String) -> Self {
            let ctx = GPUContext::new(&ContextDescriptor::default(), Some(&window)).await;

            let z_max = volume.max_per_slice();

            let z_slider = crate::util::web::get_input_element_by_id(z_slider_id.as_str());
            let z_slice = z_slider.value().parse::<i32>().unwrap();
            let z_slice_uniforms = ZSliceUniforms {
                zslice: z_slice,
                zmax: z_max[z_slice as usize] as f32,
            };
            let z_slice_uniform_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&z_slice_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });


            let tex_width = volume.shape[2];
            let tex_height = volume.shape[1];

            let volume_texture = create_volume_texture(
                volume.data.as_slice(),
                wgpu::Extent3d {
                    width: tex_width,
                    height: tex_height,
                    depth_or_array_layers: volume.shape[0]
                },
                &ctx
            );

            let (storage_texture_handle, storage_texture_view) = create_storage_texture(&ctx.device, tex_width, tex_height);
            let storage_texture = Texture {
                texture: storage_texture_handle,
                view: storage_texture_view
            };

            let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            let z_slice_pass = ZSlicePass::new(
                &volume_texture.view,
                &storage_texture.view,
                tex_width, tex_height,
                &z_slice_uniform_buffer,
                &ctx
            );
            let full_screen_pass = FullScreenPass::new(
                &storage_texture.view,
                &sampler,
                &ctx
            );

            Self {
                window,
                ctx,
                volume_texture,
                storage_texture,
                sampler,
                z_slice_pass,
                full_screen_pass,
                z_slider,
                z_slice_uniform_buffer,
                z_max,
            }
        }

        pub fn update(&self) {
            let z_slice = self.z_slider.value().parse::<i32>().unwrap();
            let uniforms = ZSliceUniforms {
                zslice: z_slice,
                zmax: self.z_max[z_slice as usize] as f32,
            };
            self.ctx.queue.write_buffer(
                &self.z_slice_uniform_buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );
        }

        pub fn render(&self, canvas_view: &wgpu::TextureView) {
            let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder = self.z_slice_pass.add_to_command(encoder);
            encoder = self.full_screen_pass.add_to_command(encoder, &canvas_view);
            self.ctx.queue.submit(Some(encoder.finish()));
        }

        pub fn run(z_slicer: Self, event_loop: EventLoop<()>) {
            event_loop.run(move |event, _, control_flow| {
                // force ownership by the closure
                let _ = (&z_slicer.ctx.instance, &z_slicer.ctx.adapter);

                *control_flow = winit::event_loop::ControlFlow::Poll;

                match event {
                    event::Event::RedrawEventsCleared => {
                        z_slicer.window.request_redraw();
                    }
                    event::Event::RedrawRequested(_) => {
                        z_slicer.update();

                        let frame = match z_slicer.ctx.surface.as_ref().unwrap().get_current_texture() {
                            Ok(frame) => frame,
                            Err(_) => {
                                z_slicer.ctx.surface.as_ref().unwrap().configure(&z_slicer.ctx.device, &z_slicer.ctx.surface_configuration.as_ref().unwrap());
                                z_slicer.ctx.surface
                                    .as_ref()
                                    .unwrap()
                                    .get_current_texture()
                                    .expect("Failed to acquire next surface texture!")
                            }
                        };
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        z_slicer.render(&view);

                        frame.present();
                    }
                    _ => {}
                }
            });
        }
    }

    pub async fn compute_to_image_test(window: &winit::window::Window) {
        fn create_mandelbrot_texels(size: usize) -> Vec<u8> {
            (0..size * size)
                .map(|id| {
                    // get high five for recognizing this ;)
                    let cx = 3.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
                    let cy = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
                    let (mut x, mut y, mut count) = (cx, cy, 0);
                    while count < 0xFF && x * x + y * y < 4.0 {
                        let old_x = x;
                        x = x * x - y * y + cx;
                        y = 2.0 * old_x * y + cy;
                        count += 1;
                    }
                    count
                })
                .collect()
        }
        fn create_input_texture(device: &wgpu::Device, mut queue: &wgpu::Queue, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
            let texels = create_mandelbrot_texels(size as usize);
            let texture_extent = wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: texture_extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            });
            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            queue.write_texture(
                texture.as_image_copy(),
                &texels,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(size).unwrap()),
                    rows_per_image: None,
                },
                texture_extent,
            );
            (texture, texture_view)
        }

        let ctx = GPUContext::new(&ContextDescriptor::default(), Some(window)).await;

        let tex_size = window.inner_size().width;
        let (input_texture, input_texture_view) = create_input_texture(&ctx.device, &ctx.queue, tex_size);
        let (storage_texture, storage_texture_view) = create_storage_texture(&ctx.device, tex_size, tex_size);
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let edge_detect_pass = EdgeDetectPass::new(&input_texture_view, &storage_texture_view, tex_size, tex_size, &ctx);
        let full_screen_pass = FullScreenPass::new(&storage_texture_view, &sampler, &ctx);

        let frame = ctx.surface.as_ref().unwrap().get_current_texture().unwrap();
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder = edge_detect_pass.add_to_command(encoder);
        encoder = full_screen_pass.add_to_command(encoder, &view);

        // Submits command encoder for processing
        ctx.queue.submit(Some(encoder.finish()));

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        ctx.device.poll(wgpu::Maintain::Wait);

        frame.present();
    }
}
