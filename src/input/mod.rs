use instant::now;

#[derive(Copy, Clone, Debug)]
#[readonly::make]
pub struct Frame {
    pub number: u32,
}

impl Frame {
    pub fn new(number: u32) -> Self {
        Self {
            number
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[readonly::make]
pub struct Time {
    pub now: f32,
    pub delta: f32,
}

impl Time {
    pub fn new(now: f32, last: f32) -> Self {
        Self {
            now,
            delta: (now - last).abs(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Input {
    pub frame: Frame,
    pub time: Time,
}

impl Input {
    pub fn new() -> Self {
        let now = now() as f32;
        Self {
            frame: Frame::new(0),
            time: Time::new(now, now),
        }
    }

    pub fn from_last(last: &Input) -> Self {
        let now = now() as f32;
        Self {
            frame: Frame::new(last.frame.number + 1),
            time: Time::new(now, last.time.now),
        }
    }
}
