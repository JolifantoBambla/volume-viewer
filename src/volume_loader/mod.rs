use std::sync::mpsc;
use zarr_wasm::group::Group;
use zarr_wasm::zarr::ZarrArray;

pub struct Channel<T, V> {
    pub receiver: mpsc::Receiver<T>,
    pub sender: mpsc::Sender<V>,
}

pub fn create_new_channel_pair<Request, Response>() -> (Channel<Response, Request>, Channel<Request, Response>) {
    let (request_sender, request_receiver) = mpsc::channel::<Request>();
    let (response_sender, response_receiver) = mpsc::channel::<Response>();
    (
        Channel::<Response, Request> {
            receiver: response_receiver,
            sender: request_sender,
        },
        Channel::<Request, Response> {
            receiver: request_receiver,
            sender: response_sender,
        },
    )
}

pub struct VolumeRequest {
    pub store: String,
    pub path: String,
}

pub struct BrickRequest {
    // todo: needs resolution level
    // todo: needs brick coordinates
}

pub enum RequestEvent {
    Volume(VolumeRequest),
    Brick(BrickRequest),
}

pub struct VolumeResponse {
    // todo: needs volume meta data
}

pub struct BrickResponse {
    // todo: needs brick data
    pub data: Vec<u8>,
    pub shape: Vec<u32>,
}

pub enum ResponseEvent {
    Volume(VolumeResponse),
    Brick(BrickResponse),
}

pub async fn test_async_loading2(sender: mpsc::Sender<ResponseEvent>) {
    let zarr_array = ZarrArray::open(
        "http://localhost:8005/".to_string(),
        "ome-zarr/m.ome.zarr/0/2".to_string(),
    ).await;
    sender.send(ResponseEvent::Brick(BrickResponse {
        data: vec![],
        shape: zarr_array.shape(),
    })).unwrap();
}

async fn open_ome_zarr(store: String, path: String, sender: mpsc::Sender<ResponseEvent>) {
    let group = Group::open(store.clone(), "ome-zarr/m.ome.zarr/0".to_string()).await;
    let array = ZarrArray::open(store, path).await;
    log::info!("{:?}", serde_json::to_string(&group.get_attributes().await));
    sender.send(ResponseEvent::Brick(BrickResponse {
        data: vec![],
        shape: array.shape(),
    })).unwrap();
}

fn using_event_loop(channel: Channel<RequestEvent, ResponseEvent>) {
    let event_loop = winit::event_loop::EventLoop::new();
    event_loop.run(move |event, _, control_flow| {
        log::info!("event handler");
        //*control_flow = winit::event_loop::ControlFlow::Poll;
        let request = channel.receiver.try_recv();
        if request.is_ok() {
            let request = request.unwrap();
            match request {
                RequestEvent::Volume(request) => {
                    wasm_bindgen_futures::spawn_local(open_ome_zarr(request.store, request.path, channel.sender.clone()));
                }
                RequestEvent::Brick(request) => {}
            }
        }
    });
}

pub async fn using_async (channel: Channel<RequestEvent, ResponseEvent>) {
    let request = channel.receiver.try_recv();
    if request.is_ok() {
        let request = request.unwrap();
        match request {
            RequestEvent::Volume(request) => {
                open_ome_zarr(request.store, request.path, channel.sender.clone()).await;
            }
            RequestEvent::Brick(request) => {}
        }
    }
}

fn using_loop(channel: Channel<RequestEvent, ResponseEvent>) {
    loop {
        log::info!("event handler");
        //*control_flow = winit::event_loop::ControlFlow::Poll;
        let request = channel.receiver.try_recv();
        if request.is_ok() {
            let request = request.unwrap();
            match request {
                RequestEvent::Volume(request) => {
                    wasm_bindgen_futures::spawn_local(open_ome_zarr(request.store, request.path, channel.sender.clone()));
                }
                RequestEvent::Brick(request) => {}
            }
        }
    }
}

pub fn load_volume(sender: mpsc::Sender<ResponseEvent>) {
    log::info!("load volume");
    let (response_sender, response_receiver) = mpsc::channel::<ResponseEvent>();
    log::info!("spawning future");
    wasm_bindgen_futures::spawn_local(test_async_loading2(response_sender.clone()));
    log::info!("awaiting response");
    let response = response_receiver.recv().expect("Expected loader to finish");
    log::info!("passing on response");
    sender.send(response).ok();

}

pub fn loader_main(channel: Channel<RequestEvent, ResponseEvent>) {
    using_loop(channel);
    //using_event_loop(channel);
    //wasm_bindgen_futures::spawn_local(using_async(channel));
}
