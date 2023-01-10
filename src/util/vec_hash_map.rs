use std::collections::hash_map::Iter;
use std::collections::HashMap;
use std::hash::Hash;

pub struct VecHashMap<K, V> {
    data: HashMap<K, Vec<V>>,
}

impl<K: Eq + Hash, V> VecHashMap<K, V> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new()
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        if !self.data.contains_key(&key) {
            self.data.insert(key, vec![value]);
        } else {
            self.data.get_mut(&key).unwrap().push(value);
        }
    }

    pub fn get(&self, key: &K) -> Option<&Vec<V>> {
        self.data.get(key)
    }

    pub fn iter(&self) -> Iter<'_, K, Vec<V>> {
        self.data.iter()
    }
}
