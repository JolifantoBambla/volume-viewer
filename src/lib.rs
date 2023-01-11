use crate::app::{App, GLOBAL_EVENT_LOOP_PROXY};
use crate::event::{Event, RawArrayReceived};
use crate::renderer::settings::MultiChannelVolumeRendererSettings;
use crate::renderer::volume::RawVolumeBlock;
use crate::util::init;
use crate::volume::BrickedMultiResolutionMultiVolumeMeta;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wgpu_framework::app::AppRunner;
use wgpu_framework::util::window::WindowConfig;

pub mod app;
pub mod event;
pub mod gpu_list;
pub mod preprocessing;
pub mod renderer;
pub mod resource;
pub mod util;
pub mod volume;
pub mod wgsl;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// This function is run when the WASM module is instantiated.
#[wasm_bindgen(start)]
pub fn initialize() {
    // note: init stuff used to happen here, but when using WebAssembly threads this function is
    // called once per thread and not once globally, so this moved to `main
}

///
#[wasm_bindgen(js_name = "dispatchChunkReceived")]
pub fn dispatch_chunk_received(data: Vec<u16>, shape: Vec<u32>) {
    unsafe {
        if let Some(event_loop_proxy) = GLOBAL_EVENT_LOOP_PROXY.as_ref() {
            event_loop_proxy
                .send_event(Event::RawArray(RawArrayReceived { data, shape }))
                .ok();
        } else {
            log::error!("dispatchChunkReceived called on uninitialized event loop");
        }
    }
}

// todo: lib.rs should contain only init and event_loop stuff
//  communication with JS side should only be done via events
//  create CustomEvent enum that consists of WindowEvent and can be extended by other events
//  create thread pool that is supplied with event proxies sending events to event loop
#[wasm_bindgen]
pub fn main(canvas: JsValue, volume_meta: JsValue, render_settings: JsValue) {
    // todo: make logger configurable
    init::set_panic_hook();
    init::set_logger(None);

    let volume_meta: BrickedMultiResolutionMultiVolumeMeta =
        serde_wasm_bindgen::from_value(volume_meta)
            .expect("Received invalid volume meta. Shutting down.");

    let render_settings: MultiChannelVolumeRendererSettings =
        serde_wasm_bindgen::from_value(render_settings)
            .expect("Received invalid render settings. Shutting down.");

    wasm_bindgen_futures::spawn_local(start_event_loop(canvas, volume_meta, render_settings));
}

async fn start_event_loop(
    canvas: JsValue,
    volume_meta: BrickedMultiResolutionMultiVolumeMeta,
    render_settings: MultiChannelVolumeRendererSettings,
) {
    let window_config = WindowConfig::new_with_offscreen_canvas(
        "Volume Viewer".to_string(),
        canvas.unchecked_into(),
    );
    let app_runner = AppRunner::<App>::new(window_config).await;
    let app = App::new(
        app_runner.ctx().gpu(),
        app_runner.window(),
        app_runner.ctx().surface_configuration(),
        volume_meta,
        render_settings,
    )
    .await;

    // NOTE: All resource allocations should happen before the main render loop
    // The reason for this is that receiving allocation errors is async, but
    let start_closure = Closure::once_into_js(move || app_runner.run(app));

    // make sure to handle JS exceptions thrown inside start_closure.
    // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
    if let Err(error) = call_catch(&start_closure) {
        web_sys::console::error_1(&error);
    }

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
        fn call_catch(this: &JsValue) -> Result<(), JsValue>;
    }
}

pub fn make_raw_volume_block(data: Vec<u16>, shape: Vec<u32>) -> RawVolumeBlock {
    let volume_max = *data.iter().max().unwrap() as f32;
    RawVolumeBlock::new(
        data.iter()
            .map(|x| ((*x as f32 / volume_max) * 255.) as u8)
            .collect(),
        volume_max as u32,
        shape[2],
        shape[1],
        shape[0],
    )
}
