struct ListU32 {
    list: array<u32>,
}

struct ListMeta {
    capacity: u32,
    fill_pointer: atomic<u32>,
}
