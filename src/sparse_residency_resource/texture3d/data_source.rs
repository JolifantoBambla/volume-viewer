use crate::sparse_residency_resource::texture3d::page_table::PageTableAddress;
use crate::sparse_residency_resource::texture3d::volume_meta::{
    BrickAddress, MultiResolutionVolumeMeta,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CustomEvent, EventTarget};

#[derive(Deserialize, Serialize)]
pub struct Brick {
    data: Vec<u8>,
    min: u8,
    max: u8,
}

pub trait SparseResidencyTexture3DSource {
    fn get_meta(&self) -> &MultiResolutionVolumeMeta;

    fn request_bricks(&mut self, brick_addresses: Vec<PageTableAddress>);

    fn poll_bricks(&mut self, limit: usize) -> Vec<(PageTableAddress, Brick)>;
}

#[cfg(target_arch = "wasm32")]
pub const BRICK_REQUEST_EVENT: &str = "data-loader:brick-request";

#[cfg(target_arch = "wasm32")]
pub const BRICK_RESPONSE_EVENT: &str = "data-loader:brick-response";

#[cfg(target_arch = "wasm32")]
#[derive(Deserialize, Serialize)]
struct BrickEvent {
    address: [u32; 4],
    brick: Brick,
}

#[cfg(target_arch = "wasm32")]
pub struct HtmlEventTargetTexture3DSource {
    volume_meta: MultiResolutionVolumeMeta,
    brick_queue: Rc<RefCell<VecDeque<(PageTableAddress, Brick)>>>,
    event_target: EventTarget,
}

#[cfg(target_arch = "wasm32")]
impl HtmlEventTargetTexture3DSource {
    pub fn new(volume_meta: MultiResolutionVolumeMeta, event_target: EventTarget) -> Self {
        let mut brick_queue = Rc::new(RefCell::new(VecDeque::new()));
        let mut receiver = brick_queue.clone();
        let event_callback = Closure::wrap(Box::new(move |event: JsValue| {
            let custom_event = event.unchecked_into::<CustomEvent>();
            let brick_event: BrickEvent =
                serde_wasm_bindgen::from_value(custom_event.detail()).expect("Invalid BrickEvent");
            receiver.borrow_mut().push_back((
                PageTableAddress::from(brick_event.address),
                brick_event.brick,
            ));
        }) as Box<dyn FnMut(JsValue)>);

        event_target
            .add_event_listener_with_callback(
                BRICK_RESPONSE_EVENT,
                event_callback.as_ref().unchecked_ref(),
            )
            .ok();
        event_callback.forget();

        Self {
            volume_meta,
            brick_queue,
            event_target,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl SparseResidencyTexture3DSource for HtmlEventTargetTexture3DSource {
    fn get_meta(&self) -> &MultiResolutionVolumeMeta {
        &self.volume_meta
    }

    fn request_bricks(&mut self, brick_addresses: Vec<PageTableAddress>) {
        let request_event = web_sys::CustomEvent::new(BRICK_REQUEST_EVENT).ok().unwrap();
        let mut request_data: HashMap<&str, Vec<BrickAddress>> = HashMap::new();
        request_data.insert(
            "addresses",
            brick_addresses
                .iter()
                .map(|&a| BrickAddress {
                    index: a.location.to_array(),
                    level: a.level,
                    channel: 0,
                })
                .collect(),
        );
        request_event.init_custom_event_with_can_bubble_and_cancelable_and_detail(
            BRICK_REQUEST_EVENT,
            false,
            false,
            &JsValue::from_serde(&request_data).unwrap(),
        );
        self.event_target.dispatch_event(&request_event).ok();
    }

    fn poll_bricks(&mut self, limit: usize) -> Vec<(PageTableAddress, Brick)> {
        self.brick_queue.borrow_mut().drain(..limit).collect()
    }
}
