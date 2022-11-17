use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CustomEvent, EventTarget};

use crate::volume::{Brick, BrickAddress, BrickedMultiResolutionMultiVolumeMeta};

/// A data source providing access to a bricked multi-resolution multi-channel volume data source.
pub trait VolumeDataSource {
    fn get_meta(&self) -> &BrickedMultiResolutionMultiVolumeMeta;

    /// Stores a `brick` and it's `address` such that it may later be polled using `VolumeDataSource::poll_bricks`.
    ///
    /// # Arguments
    ///
    /// * `address`: the address of the given brick in the multi-resolution multi-channel volume.
    /// * `brick`: the brick
    ///
    /// returns: ()
    fn receive_brick(&mut self, address: BrickAddress, brick: Brick);

    /// Requests the bricks at the given `brick_addresses`, their addresses in the multi-resolutions multi-channel volume.
    ///
    /// # Arguments
    ///
    /// * `brick_addresses`: the addresses of the bricks to request from the underlying data source.
    ///
    /// returns: ()
    fn request_bricks(&mut self, brick_addresses: Vec<BrickAddress>);

    /// Receives up to `limit` bricks stored in this data source and their addresses in the multi-resolution multi-channel volume.
    /// If the returned bricks are in the same order as they have been received is implementation-dependent.
    ///
    /// # Arguments
    ///
    /// * `limit`: the maximum number of bricks to poll from this data source.
    ///
    /// returns: Vec<(BrickAddress, Brick), Global>
    fn poll_bricks(&mut self, limit: usize) -> Vec<(BrickAddress, Brick)>;
}

#[cfg(target_arch = "wasm32")]
#[derive(Deserialize, Serialize)]
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
pub struct HtmlEventTargetVolumeDataSource {
    volume_meta: BrickedMultiResolutionMultiVolumeMeta,
    brick_queue: Rc<RefCell<VecDeque<(BrickAddress, Brick)>>>,
    event_target: EventTarget,
}

#[cfg(target_arch = "wasm32")]
impl HtmlEventTargetVolumeDataSource {
    pub fn new(
        volume_meta: BrickedMultiResolutionMultiVolumeMeta,
        event_target: EventTarget,
    ) -> Self {
        let brick_queue = Rc::new(RefCell::new(VecDeque::new()));
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
            event_target,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl VolumeDataSource for HtmlEventTargetVolumeDataSource {
    fn get_meta(&self) -> &BrickedMultiResolutionMultiVolumeMeta {
        &self.volume_meta
    }

    fn receive_brick(&mut self, address: BrickAddress, brick: Brick) {
        self.brick_queue.borrow_mut().push_back((address, brick));
    }

    fn request_bricks(&mut self, brick_addresses: Vec<BrickAddress>) {
        let request_event = web_sys::CustomEvent::new(BRICK_REQUEST_EVENT).ok().unwrap();
        let mut request_data: HashMap<&str, Vec<BrickAddress>> = HashMap::new();
        request_data.insert("addresses", brick_addresses);
        request_event.init_custom_event_with_can_bubble_and_cancelable_and_detail(
            BRICK_REQUEST_EVENT,
            false,
            false,
            &JsValue::from_serde(&request_data).unwrap(),
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
}
