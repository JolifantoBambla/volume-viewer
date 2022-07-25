pub fn u32s_to_be_bytes(numbers: &[u32]) -> Vec<u8> {
    numbers.iter().flat_map(|val| val.to_be_bytes()).collect()
}

pub fn u32s_to_le_bytes(numbers: &[u32]) -> Vec<u8> {
    numbers.iter().flat_map(|val| val.to_le_bytes()).collect()
}

pub fn u32s_to_ne_bytes(numbers: &[u32]) -> Vec<u8> {
    numbers.iter().flat_map(|val| val.to_ne_bytes()).collect()
}
