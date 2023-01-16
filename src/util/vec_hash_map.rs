use std::collections::btree_map::{Entry, Iter};
use std::collections::BTreeMap;
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct VecHashMap<K, V> {
    data: BTreeMap<K, Vec<V>>,
}

impl<K: Copy + Eq + Hash + Ord, V> VecHashMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: K, value: V) {
        if let Entry::Vacant(e) = self.data.entry(key) {
            e.insert(vec![value]);
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

impl<K, V> Default for VecHashMap<K, V> {
    fn default() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }
}
