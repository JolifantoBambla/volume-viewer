use std::sync::mpsc;

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


