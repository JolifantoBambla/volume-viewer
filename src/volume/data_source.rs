use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::rc::Rc;
use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CustomEvent, EventTarget};

use crate::volume::{Brick, BrickAddress, BrickedMultiResolutionMultiVolumeMeta};

pub trait VolumeDataSource: Debug {
    fn meta(&self) -> &BrickedMultiResolutionMultiVolumeMeta;

    fn enqueue_brick(&mut self, brick: (BrickAddress, Brick));

    fn request_bricks(&mut self, brick_addresses: Vec<BrickAddress>);

    fn poll_bricks(&mut self, limit: usize) -> Vec<(BrickAddress, Brick)>;

    #[cfg(target_arch = "wasm32")]
    fn poll_bricks_unchecked(&mut self, limit: usize) -> Vec<Rc<(BrickAddress, Uint8Array)>>;

    #[cfg(target_arch = "wasm32")]
    fn enqueue_brick_unchecked(&mut self, brick: Rc<(BrickAddress, Uint8Array)>);
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Deserialize, Serialize)]
struct BrickEvent {
    address: BrickAddress,
    brick: Brick,
}

#[cfg(target_arch = "wasm32")]
pub const BRICK_REQUEST_EVENT: &str = "data-loader:brick-request";

#[cfg(target_arch = "wasm32")]
pub const BRICK_RESPONSE_EVENT: &str = "data-loader:brick-response";

/// A `SparseResidencyTexture3DSource` that is backed by an `web_sys::EventTarget`.
/// It uses the `web_sys::EventTarget` to pass on brick requests to and receive brick responses from
/// it.
/// It is agnostic of the way the `web_sys::EventTarget` actually acquires bricks.
#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
pub struct HtmlEventTargetVolumeDataSource {
    volume_meta: BrickedMultiResolutionMultiVolumeMeta,
    brick_queue: Rc<RefCell<VecDeque<(BrickAddress, Brick)>>>,
    brick_queue_unchecked: Rc<RefCell<VecDeque<Rc<(BrickAddress, Uint8Array)>>>>,
    event_target: EventTarget,
}

#[cfg(target_arch = "wasm32")]
impl HtmlEventTargetVolumeDataSource {
    pub fn new(
        volume_meta: BrickedMultiResolutionMultiVolumeMeta,
        event_target: EventTarget,
    ) -> Self {
        let brick_queue = Rc::new(RefCell::new(VecDeque::new()));
        let brick_queue_unchecked = Rc::new(RefCell::new(VecDeque::new()));
        let receiver = brick_queue.clone();

        let event_callback = Closure::wrap(Box::new(move |event: JsValue| {
            let custom_event = event.unchecked_into::<CustomEvent>();
            let brick_event: BrickEvent =
                serde_wasm_bindgen::from_value(custom_event.detail()).expect("Invalid BrickEvent");
            receiver
                .borrow_mut()
                .push_back((brick_event.address, brick_event.brick));
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
            brick_queue_unchecked,
            event_target,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl VolumeDataSource for HtmlEventTargetVolumeDataSource {
    fn meta(&self) -> &BrickedMultiResolutionMultiVolumeMeta {
        &self.volume_meta
    }

    fn enqueue_brick(&mut self, brick: (BrickAddress, Brick)) {
        self.brick_queue.borrow_mut().push_back(brick);
    }

    fn request_bricks(&mut self, brick_addresses: Vec<BrickAddress>) {
        let request_event = web_sys::CustomEvent::new(BRICK_REQUEST_EVENT).ok().unwrap();
        let mut request_data: HashMap<&str, Vec<BrickAddress>> = HashMap::new();
        request_data.insert("addresses", brick_addresses);
        request_event.init_custom_event_with_can_bubble_and_cancelable_and_detail(
            BRICK_REQUEST_EVENT,
            false,
            false,
            &serde_wasm_bindgen::to_value(&request_data).expect("Could not serialize request data"),
        );
        self.event_target.dispatch_event(&request_event).ok();
    }

    fn poll_bricks(&mut self, limit: usize) -> Vec<(BrickAddress, Brick)> {
        let bricks_in_queue = self.brick_queue.borrow().len();
        self.brick_queue
            .borrow_mut()
            .drain(..min(limit, bricks_in_queue))
            .collect()
    }

    fn poll_bricks_unchecked(&mut self, limit: usize) -> Vec<Rc<(BrickAddress, Uint8Array)>> {
        let bricks_in_queue = self.brick_queue_unchecked.borrow().len();
        self.brick_queue_unchecked
            .borrow_mut()
            .drain(..min(limit, bricks_in_queue))
            .collect()
    }

    fn enqueue_brick_unchecked(&mut self, brick: Rc<(BrickAddress, Uint8Array)>) {
        self.brick_queue_unchecked.borrow_mut().push_back(brick);
    }
}
