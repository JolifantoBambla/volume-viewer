pub mod camera;
pub mod context;
pub mod geometry;
pub mod pass;
pub mod passes;
pub mod resources;
pub mod volume;

pub mod offscreen_playground {
    use crate::renderer::camera::{Camera, CameraView, Projection};
    use crate::renderer::context::{ContextDescriptor, GPUContext};
    use crate::renderer::geometry::Bounds3D;
    use crate::renderer::pass::GPUPass;
    use crate::renderer::passes::{dvr, present_to_screen};
    use crate::renderer::resources;
    use crate::renderer::volume::RawVolumeBlock;
    use bytemuck;
    use glam::{Vec2, Vec3};
    use std::sync::Arc;
    use web_sys::OffscreenCanvas;
    use wgpu::util::DeviceExt;

    pub struct DVR {
        canvas: OffscreenCanvas,
        ctx: Arc<GPUContext>,

        dvr_pass: dvr::DVR,
        present_to_screen: present_to_screen::PresentToScreen,

        dvr_bind_group: wgpu::BindGroup,
        present_to_screen_bind_group: wgpu::BindGroup,

        dvr_result_extent: wgpu::Extent3d,

        volume_transform: glam::Mat4,
        uniform_buffer: wgpu::Buffer,
    }

    impl DVR {
        pub async fn new(canvas: OffscreenCanvas, volume: RawVolumeBlock) -> Self {
            let ctx = Arc::new(
                GPUContext::new(&ContextDescriptor::default())
                    .await
                    .with_surface_from_offscreen_canvas(&canvas)
            );

            let volume_texture =
                resources::Texture::from_raw_volume_block(&ctx.device, &ctx.queue, &volume);
            let storage_texture = resources::Texture::create_storage_texture(
                &ctx.device,
                canvas.width(),
                canvas.height(),
            );

            let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            // the volume is a unit cube ([0,1]^3)
            // we translate it s.t. its center is the origin and scale it to its original dimensions
            // todo: scale should come from volume meta data (-> todo: add meta data to volume)
            let volume_transform = glam::Mat4::from_scale(volume.create_vec3()).mul_mat4(
                &glam::Mat4::from_translation(glam::Vec3::new(-0.5, -0.5, -0.5)),
            );

            let uniforms = dvr::Uniforms {
                world_to_object: volume_transform,
                ..Default::default()
            };
            let uniform_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let z_slice_pass = dvr::DVR::new(&ctx);
            let full_screen_pass = present_to_screen::PresentToScreen::new(&ctx);
            let z_slice_bind_group = z_slice_pass.create_bind_group(dvr::Resources {
                volume: &volume_texture.view,
                volume_sampler: &sampler,
                output: &storage_texture.view,
                uniforms: &uniform_buffer,
            });
            let full_screen_bind_group =
                full_screen_pass.create_bind_group(present_to_screen::Resources {
                    sampler: &sampler,
                    source_texture: &storage_texture.view,
                });

            let dvr_result_extent = wgpu::Extent3d {
                width: canvas.width(),
                height: canvas.height(),
                depth_or_array_layers: 1,
            };

            Self {
                canvas,
                ctx,
                dvr_pass: z_slice_pass,
                dvr_bind_group: z_slice_bind_group,
                present_to_screen: full_screen_pass,
                present_to_screen_bind_group: full_screen_bind_group,
                dvr_result_extent,
                volume_transform,
                uniform_buffer,
            }
        }

        pub fn update(&self, camera: &Camera) {
            let uniforms = dvr::Uniforms::new(camera.create_uniform(), self.volume_transform);
            self.ctx
                .queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        pub fn render(&self, canvas_view: &wgpu::TextureView) {
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.dvr_pass
                .encode(&mut encoder, &self.dvr_bind_group, &self.dvr_result_extent);
            self.present_to_screen.encode(
                &mut encoder,
                &self.present_to_screen_bind_group,
                canvas_view,
            );
            self.ctx.queue.submit(Some(encoder.finish()));
        }

        pub fn run(&self) {
            let distance_from_center = 50.;

            let resolution = Vec2::new(
                self.canvas.width() as f32,
                self.canvas.height() as f32,
            );

            const TRANSLATION_SPEED: f32 = 5.0;

            const NEAR: f32 = 0.0001;
            const FAR: f32 = 1000.0;
            let perspective = Projection::new_perspective(
                f32::to_radians(45.),
                self.canvas.width() as f32 / self.canvas.height() as f32,
                NEAR,
                FAR,
            );
            let orthographic = Projection::new_orthographic(Bounds3D::new(
                (resolution * -0.5).extend(NEAR),
                (resolution * 0.5).extend(FAR),
            ));

            let mut camera = Camera::new(
                CameraView::new(
                    Vec3::new(1., 1., 1.) * distance_from_center,
                    Vec3::new(0., 0., 0.),
                    Vec3::new(0., 1., 0.),
                ),
                perspective.clone(),
            );
            let mut last_mouse_position = Vec2::new(0., 0.);
            let mut left_mouse_pressed = false;
            let mut right_mouse_pressed = false;

            loop {
                self.update(&camera);

                let frame = match self.ctx.surface.as_ref().unwrap().get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        self.ctx.surface.as_ref().unwrap().configure(
                            &self.ctx.device,
                            self.ctx.surface_configuration.as_ref().unwrap(),
                        );
                        self.ctx
                            .surface
                            .as_ref()
                            .unwrap()
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                self.render(&view);

                frame.present();
                log::info!("Frame rendered, {}, {}", self.canvas.width(), self.canvas.height());
            }
        }
    }
}


// todo: refactor this into a proper module
// this is just a small module where I test stuff
pub mod dvr_playground {
    use crate::renderer::camera::{Camera, CameraView, Projection};
    use crate::renderer::context::{ContextDescriptor, GPUContext};
    use crate::renderer::geometry::Bounds3D;
    use crate::renderer::pass::GPUPass;
    use crate::renderer::passes::{dvr, present_to_screen};
    use crate::renderer::resources;
    use crate::renderer::volume::RawVolumeBlock;
    use bytemuck;
    use glam::{Vec2, Vec3};
    use std::sync::Arc;
    use wgpu::util::DeviceExt;
    use winit::{
        event::{
            ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
            WindowEvent,
        },
        event_loop::EventLoop,
        window::Window,
    };

    pub struct DVR {
        window: winit::window::Window,
        ctx: Arc<GPUContext>,

        dvr_pass: dvr::DVR,
        present_to_screen: present_to_screen::PresentToScreen,

        dvr_bind_group: wgpu::BindGroup,
        present_to_screen_bind_group: wgpu::BindGroup,

        dvr_result_extent: wgpu::Extent3d,

        volume_transform: glam::Mat4,
        uniform_buffer: wgpu::Buffer,
    }

    impl DVR {
        pub async fn new(window: Window, volume: RawVolumeBlock) -> Self {
            let ctx = Arc::new(
                GPUContext::new(&ContextDescriptor::default())
                    .await
                    .with_surface_from_window(&window)
            );

            let volume_texture =
                resources::Texture::from_raw_volume_block(&ctx.device, &ctx.queue, &volume);
            let storage_texture = resources::Texture::create_storage_texture(
                &ctx.device,
                window.inner_size().width,
                window.inner_size().height,
            );

            let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            // the volume is a unit cube ([0,1]^3)
            // we translate it s.t. its center is the origin and scale it to its original dimensions
            // todo: scale should come from volume meta data (-> todo: add meta data to volume)
            let volume_transform = glam::Mat4::from_scale(volume.create_vec3()).mul_mat4(
                &glam::Mat4::from_translation(glam::Vec3::new(-0.5, -0.5, -0.5)),
            );

            let uniforms = dvr::Uniforms {
                world_to_object: volume_transform,
                ..Default::default()
            };
            let uniform_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let z_slice_pass = dvr::DVR::new(&ctx);
            let full_screen_pass = present_to_screen::PresentToScreen::new(&ctx);
            let z_slice_bind_group = z_slice_pass.create_bind_group(dvr::Resources {
                volume: &volume_texture.view,
                volume_sampler: &sampler,
                output: &storage_texture.view,
                uniforms: &uniform_buffer,
            });
            let full_screen_bind_group =
                full_screen_pass.create_bind_group(present_to_screen::Resources {
                    sampler: &sampler,
                    source_texture: &storage_texture.view,
                });

            let dvr_result_extent = wgpu::Extent3d {
                width: window.inner_size().width,
                height: window.inner_size().height,
                depth_or_array_layers: 1,
            };

            Self {
                window,
                ctx,
                dvr_pass: z_slice_pass,
                dvr_bind_group: z_slice_bind_group,
                present_to_screen: full_screen_pass,
                present_to_screen_bind_group: full_screen_bind_group,
                dvr_result_extent,
                volume_transform,
                uniform_buffer,
            }
        }

        pub fn update(&self, camera: &Camera) {
            let uniforms = dvr::Uniforms::new(camera.create_uniform(), self.volume_transform);
            self.ctx
                .queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        pub fn render(&self, canvas_view: &wgpu::TextureView) {
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.dvr_pass
                .encode(&mut encoder, &self.dvr_bind_group, &self.dvr_result_extent);
            self.present_to_screen.encode(
                &mut encoder,
                &self.present_to_screen_bind_group,
                canvas_view,
            );
            self.ctx.queue.submit(Some(encoder.finish()));
        }

        pub fn run(dvr: Self, event_loop: EventLoop<()>) {
            let distance_from_center = 50.;

            let resolution = Vec2::new(
                dvr.window.inner_size().width as f32,
                dvr.window.inner_size().height as f32,
            );

            const TRANSLATION_SPEED: f32 = 5.0;

            const NEAR: f32 = 0.0001;
            const FAR: f32 = 1000.0;
            let perspective = Projection::new_perspective(
                f32::to_radians(45.),
                dvr.window.inner_size().width as f32 / dvr.window.inner_size().height as f32,
                NEAR,
                FAR,
            );
            let orthographic = Projection::new_orthographic(Bounds3D::new(
                (resolution * -0.5).extend(NEAR),
                (resolution * 0.5).extend(FAR),
            ));

            let mut camera = Camera::new(
                CameraView::new(
                    Vec3::new(1., 1., 1.) * distance_from_center,
                    Vec3::new(0., 0., 0.),
                    Vec3::new(0., 1., 0.),
                ),
                perspective.clone(),
            );
            let mut last_mouse_position = Vec2::new(0., 0.);
            let mut left_mouse_pressed = false;
            let mut right_mouse_pressed = false;

            event_loop.run(move |event, _, control_flow| {
                // force ownership by the closure
                let _ = (&dvr.ctx.instance, &dvr.ctx.adapter);

                *control_flow = winit::event_loop::ControlFlow::Poll;

                // todo: refactor input handling
                match event {
                    Event::RedrawEventsCleared => {
                        dvr.window.request_redraw();
                    }
                    Event::WindowEvent {
                        event:
                            WindowEvent::MouseWheel {
                                delta: MouseScrollDelta::PixelDelta(delta),
                                ..
                            },
                        ..
                    } => {
                        camera.view.move_forward(
                            (f64::min(delta.y.abs(), 1.) * delta.y.signum()) as f32
                                * TRANSLATION_SPEED,
                        );
                    }
                    Event::WindowEvent {
                        event: WindowEvent::MouseInput { state, button, .. },
                        ..
                    } => match button {
                        MouseButton::Left => {
                            left_mouse_pressed = state == ElementState::Pressed;
                        }
                        MouseButton::Right => {
                            right_mouse_pressed = state == ElementState::Pressed;
                        }
                        _ => {}
                    },
                    Event::WindowEvent {
                        event:
                            WindowEvent::KeyboardInput {
                                input:
                                    KeyboardInput {
                                        virtual_keycode: Some(virtual_keycode),
                                        state: ElementState::Pressed,
                                        ..
                                    },
                                ..
                            },
                        ..
                    } => match virtual_keycode {
                        VirtualKeyCode::D => camera.view.move_right(TRANSLATION_SPEED),
                        VirtualKeyCode::A => camera.view.move_left(TRANSLATION_SPEED),
                        VirtualKeyCode::W | VirtualKeyCode::Up => {
                            camera.view.move_forward(TRANSLATION_SPEED)
                        }
                        VirtualKeyCode::S | VirtualKeyCode::Down => {
                            camera.view.move_backward(TRANSLATION_SPEED)
                        }
                        VirtualKeyCode::C => {
                            if camera.projection().is_orthographic() {
                                camera.set_projection(perspective.clone());
                            } else {
                                camera.set_projection(orthographic.clone());
                            }
                        }
                        _ => {}
                    },
                    Event::WindowEvent {
                        event: WindowEvent::CursorMoved { position, .. },
                        ..
                    } => {
                        let mouse_position = glam::Vec2::new(position.x as f32, position.y as f32);
                        let delta = (mouse_position - last_mouse_position) / resolution;
                        last_mouse_position = mouse_position;

                        if left_mouse_pressed {
                            camera.view.orbit(delta, false);
                        } else if right_mouse_pressed {
                            let translation = delta * TRANSLATION_SPEED * 20.;
                            camera.view.move_right(translation.x);
                            camera.view.move_down(translation.y);
                        }
                    }
                    Event::RedrawRequested(_) => {
                        dvr.update(&camera);

                        let frame = match dvr.ctx.surface.as_ref().unwrap().get_current_texture() {
                            Ok(frame) => frame,
                            Err(_) => {
                                dvr.ctx.surface.as_ref().unwrap().configure(
                                    &dvr.ctx.device,
                                    dvr.ctx.surface_configuration.as_ref().unwrap(),
                                );
                                dvr.ctx
                                    .surface
                                    .as_ref()
                                    .unwrap()
                                    .get_current_texture()
                                    .expect("Failed to acquire next surface texture!")
                            }
                        };
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        dvr.render(&view);

                        frame.present();
                    }
                    _ => {}
                }
            });
        }
    }
}
