use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Image {
    pub path: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquisition: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct Well {
    pub images: Vec<Image>,

    pub version: String,
}
