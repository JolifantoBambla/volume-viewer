use serde::{Serialize, Deserialize};
use serde_json::Value;

#[derive(Serialize, Deserialize)]
pub struct Dataset {
    pub path: String,
}

#[derive(Serialize, Deserialize)]
pub struct Multiscale {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    pub datasets: Vec<Dataset>,

    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downscaling_type: Option<String>,

    // fields in metadata depend on `downscaling_type`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}
