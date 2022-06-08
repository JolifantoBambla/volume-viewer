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

/// Holds all access points to
pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: Option<wgpu::Surface>,
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

        Self {
            instance,
            adapter,
            device,
            queue,
            surface,
        }
    }
}

pub async fn compute_image(window: &winit::window::Window) {
    use std::{borrow::Cow, str::FromStr};
    use bytemuck;
    use wgpu::util::DeviceExt;

    fn create_texels(size: usize) -> Vec<u8> {
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
    fn create_texture(device: &wgpu::Device, mut queue: &wgpu::Queue, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texels = create_texels(size as usize);
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
    fn create_storage_texture(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, texture_view)
    }

    let tex_size = window.inner_size().width;
    let ctx = GPUContext::new(&ContextDescriptor::default(), Some(window)).await;

    let surface_configuration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: ctx.surface.as_ref().unwrap().get_preferred_format(&ctx.adapter).unwrap(),
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: wgpu::PresentMode::Mailbox,
    };
    ctx.surface.as_ref().unwrap().configure(&ctx.device, &surface_configuration);

    let (input_texture, input_texture_view) = create_texture(&ctx.device, &ctx.queue, tex_size);
    let (storage_texture, storage_texture_view) = create_storage_texture(&ctx.device, tex_size);
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let cs_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("edgedetect.wgsl"))),
    });
    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
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
                resource: wgpu::BindingResource::TextureView(&storage_texture_view),
            }
        ],
    });

    let full_screen_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("full_screen_quad.wgsl"))),
    });

    /*
    let full_screen_bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: Default::default(),
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false
                },
                count: None,
            }
        ]
    });
    let full_screen_pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&full_screen_bind_group_layout],
        push_constant_ranges: &[],
    });
    */
    let render_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None, //Some(&full_screen_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &full_screen_module,
            entry_point: "vert_main",
            buffers: &[]
        },
        fragment: Some(wgpu::FragmentState {
            module: &full_screen_module,
            entry_point: "frag_main",
            targets: &[surface_configuration.format.into()],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview: None
    });
    let full_screen_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &render_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&storage_texture_view),
            }
        ]
    });

    let frame =ctx.surface.as_ref().unwrap().get_current_texture().unwrap();
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(tex_size / 16, tex_size / 16, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &view,
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
        rpass.set_pipeline(&render_pipeline);
        rpass.set_bind_group(0, &full_screen_bind_group, &[]);
        rpass.draw(0..6, 0..1,);
    }

    // Submits command encoder for processing
    ctx.queue.submit(Some(encoder.finish()));

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    ctx.device.poll(wgpu::Maintain::Wait);

    frame.present();

    log::info!("finished rendering");
}

pub async fn compute_example() {
    use std::{borrow::Cow, str::FromStr};
    use bytemuck;
    use wgpu::util::DeviceExt;

    let ctx = GPUContext::new(&ContextDescriptor::default(), None).await;

    // set up buffer
    let numbers = vec![1, 2, 3, 4];
    log::info!("numbers: {:?}", numbers);
    // Gets the size in bytes of the buffer.
    let slice_size = numbers.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&numbers),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // set up pipeline
    let cs_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    // Instantiates the pipeline.
    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });
    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    // Submits command encoder for processing
    ctx.queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    // Gets the future representing when `staging_buffer` can be read from
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    ctx.device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    let result: Option<Vec<u32>> = if let Ok(()) = buffer_future.await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
        // If you are familiar with C++ these 2 lines can be thought of similarly to:
        //   delete myPointer;
        //   myPointer = NULL;
        // It effectively frees the memory

        // Returns data from buffer
        Some(result)
    } else {
        None
    };
    log::info!("result {:?}", result);
}
