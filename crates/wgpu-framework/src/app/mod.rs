use std::marker::PhantomData;
use wgpu::{SubmissionIndex, TextureView};
use winit::event_loop::EventLoopBuilder;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use crate::context::{ContextDescriptor, SurfaceContext, SurfaceTarget, WgpuContext};
use crate::event::lifecycle::{OnCommandsSubmitted, PrepareRender, Update};
use crate::event::window::{OnResize, OnUserEvent, OnWindowEvent};
use crate::input::Input;
#[cfg(target_arch = "wasm32")]
use crate::util::web::get_or_create_window;
use crate::util::window::{CanvasConfig, WindowConfig};

pub trait MapToWindowEvent: OnUserEvent {
    /// Maps an instance of `Self::UserEvent` to a `WindowEvent`.
    /// This is a work-around to move a `WindowEvent` out of a custom event type and pass it to an
    /// `Input` object.
    /// This makes sense in a web worker context where `WindowEvent`s can not be passed to the
    /// `EventLoop` from an `web_sys::OffscreenCanvas` because the base types that are usually
    /// deserialized to a `WindowEvent` do not exist in web workers.
    /// In all other cases this can just return `None`.
    fn map_to_window_event(&self, user_event: &Self::UserEvent) -> Option<WindowEvent>;
}

pub trait GpuApp: OnUserEvent + PrepareRender + Update + OnCommandsSubmitted {
    fn init(
        &mut self,
        window: &Window,
        event_loop: &EventLoop<Self::UserEvent>,
        context: &SurfaceContext,
    );
    fn render(&mut self, view: &TextureView, input: &Input) -> SubmissionIndex;

    fn get_context_descriptor() -> ContextDescriptor<'static>;
}

pub struct AppRunner<G: 'static + GpuApp + OnResize + OnWindowEvent + MapToWindowEvent> {
    ctx: WgpuContext,
    event_loop: Option<EventLoop<G::UserEvent>>,
    window: Window,
    phantom_data: PhantomData<G>,
}

impl<G: 'static + GpuApp + OnResize + OnWindowEvent + MapToWindowEvent> AppRunner<G> {
    #[cfg(target_arch = "wasm32")]
    pub async fn new(window_config: WindowConfig) -> Self {
        let event_loop = EventLoopBuilder::<G::UserEvent>::with_user_event().build();
        let window = get_or_create_window(&window_config, &event_loop);

        let surface_target = match window_config.canvas_config() {
            CanvasConfig::OffscreenCanvas(offscreen_canvas) => {
                SurfaceTarget::OffscreenCanvas(offscreen_canvas)
            }
            _ => SurfaceTarget::Window(&window),
        };

        let context = WgpuContext::new(&G::get_context_descriptor(), Some(surface_target)).await;

        AppRunner {
            ctx: context,
            event_loop: Some(event_loop),
            window,
            phantom_data: PhantomData,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn run(mut self, mut app: G) {
        let mut input = Input::new(
            self.ctx().surface_configuration().width,
            self.ctx().surface_configuration().height,
        );

        let event_loop = self.event_loop.take().unwrap();

        app.init(&self.window, &event_loop, self.ctx());

        log::debug!("Starting event loop");
        event_loop.spawn(move |event, _, control_flow| {
            let _ = (self.ctx.instance(), self.ctx.adapter());

            *control_flow = ControlFlow::Poll;

            match event {
                event::Event::RedrawEventsCleared => {
                    self.window.request_redraw();
                }
                event::Event::WindowEvent {
                    event:
                        WindowEvent::Resized(size)
                        | WindowEvent::ScaleFactorChanged {
                            new_inner_size: &mut size,
                            ..
                        },
                    ..
                } => {
                    log::debug!("Resizing to {:?}", size);
                    let width = size.width.max(1);
                    let height = size.height.max(1);
                    self.ctx.surface_context_mut().on_resize(width, height);
                    app.on_resize(width, height);
                    input.on_resize(width, height);
                }
                event::Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input:
                            event::KeyboardInput {
                                virtual_keycode: Some(event::VirtualKeyCode::Escape),
                                state: event::ElementState::Pressed,
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {
                        input.on_window_event(&event);
                        app.on_window_event(&event);
                    }
                },
                event::Event::UserEvent(e) => {
                    app.on_user_event(&e);
                    let window_event = app.map_to_window_event(&e);
                    if let Some(window_event) = window_event {
                        input.on_window_event(&window_event);
                    }
                }
                event::Event::RedrawRequested(_) => {
                    let frame_input = input.prepare_next();
                    app.update(&frame_input);

                    app.prepare_render(&frame_input);

                    let frame = match self.ctx().surface().get_current_texture() {
                        Ok(frame) => frame,
                        Err(_) => {
                            self.ctx().configure_surface();
                            self.ctx()
                                .surface()
                                .get_current_texture()
                                .expect("Failed to acquire next surface texture!")
                        }
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let submission_index = app.render(&view, &frame_input);
                    app.on_commands_submitted(&frame_input, &submission_index);

                    frame.present();
                }
                _ => {}
            }
        });
    }

    pub fn event_loop(&self) -> &Option<EventLoop<G::UserEvent>> {
        &self.event_loop
    }
    pub fn ctx(&self) -> &SurfaceContext {
        self.ctx.surface_context()
    }
    pub fn window(&self) -> &Window {
        &self.window
    }
}
