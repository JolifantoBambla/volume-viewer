struct ListU32 {
    list: array<u32>,
}

struct ListMeta {
    capacity: u32,
    fill_pointer: atomic<u32>,
    written_at: u32,
}

struct ReadOnlyListMeta {
    capacity: u32,
    fill_pointer: u32,
    written_at: u32,
}
