#[derive(Copy, Clone, Debug, Default)]
pub struct Frame {
    number: u32,
}

impl Frame {
    pub fn next(&self) -> Self {
        Self {
            number: self.number + 1,
        }
    }

    pub fn number(&self) -> u32 {
        self.number
    }
}
