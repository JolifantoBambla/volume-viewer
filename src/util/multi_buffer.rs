pub struct MultiBuffered<T> {
    resources: Vec<T>,
}

impl<T> MultiBuffered<T> {
    pub fn new<F: Fn() -> T>(constructor: F, capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            resources: (0..capacity).map(|_| constructor()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn get(&self, index: usize) -> &T {
        let i = index % self.len();
        &self.resources[i]
    }

    pub fn get_mut(&mut self, index: usize) -> &T {
        let i = index % self.len();
        self.resources.get_mut(i).unwrap()
    }

    pub fn get_previous(&self, index: usize) -> &T {
        let i = if index == 0 {
            index + self.len() - 1
        } else {
            index - 1
        };
        self.get(i)
    }

    pub fn get_previous_mut(&mut self, index: usize) -> &T {
        let i = if index == 0 {
            index + self.len() - 1
        } else {
            index - 1
        };
        self.get_mut(i)
    }

    pub fn get_next(&self, index: usize) -> &T {
        let i = index + 1;
        self.get(i)
    }

    pub fn get_next_mut(&mut self, index: usize) -> &T {
        let i = index + 1;
        self.get_mut(i)
    }
}
