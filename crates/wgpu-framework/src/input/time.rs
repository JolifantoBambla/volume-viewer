use instant::now;

#[derive(Copy, Clone, Debug)]
pub struct Time {
    start: f32,
    now: f32,
    delta: f32,
}

impl Time {
    pub fn start(&self) -> f32 {
        self.start
    }
    pub fn now(&self) -> f32 {
        self.now
    }
    pub fn delta(&self) -> f32 {
        self.delta
    }
    pub fn total(&self) -> f32 {
        self.now - self.start
    }
    pub fn next(&self) -> Self {
        let now = now() as f32;
        Self {
            start: now,
            now,
            delta: now - self.now,
        }
    }
}

impl Default for Time {
    fn default() -> Self {
        let now = now() as f32;
        Self {
            start: now,
            now,
            delta: 0.0,
        }
    }
}
