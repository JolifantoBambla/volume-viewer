use serde::{Serialize, Deserialize};

// todo: check out https://docs.openmicroscopy.org/omero/5.6.1/developers/Web/WebGateway.html#imgdata
#[derive(Serialize, Deserialize)]
pub struct Window {
    pub end: i32,
    pub max: i32,
    pub min: i32,
    pub start: i32,
}

#[derive(Serialize, Deserialize)]
pub struct Channel {
    pub active: bool,
    pub coefficient: i32,
    pub color: String,
    pub family: String,
    pub inverted: bool,
    pub label: String,
    pub window: Window,
}

#[derive(Serialize, Deserialize)]
pub struct RDefs {
    pub default_t: i32,
    pub default_z: i32,
    pub model: String,
}

#[derive(Serialize, Deserialize)]
pub struct Omero {
    pub id: i32,

    pub name: String,

    pub channels: Vec<Channel>,

    #[serde(rename = "rdefs")]
    pub r_defs: RDefs,
}
