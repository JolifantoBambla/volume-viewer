use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Acquisition {
    pub id: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(rename = "maximumfieldcount")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_field_count: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(rename = "starttime")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<u64>,

    #[serde(rename = "endtime")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<u64>,
}

#[derive(Serialize, Deserialize)]
pub struct Column {
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct Row {
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct Well {
    pub path: String,
}

#[derive(Serialize, Deserialize)]
pub struct Plate {
    pub name: String,

    // [sic!]
    pub field_count: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquisitions: Option<Vec<Acquisition>>,

    pub columns: Vec<Column>,

    pub rows: Vec<Row>,

    pub wells: Vec<Well>,
}
