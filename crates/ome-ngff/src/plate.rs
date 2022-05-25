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

    #[serde(rename = "rowIndex")]
    pub row_index: u64,

    #[serde(rename = "columnIndex")]
    pub column_index: u64,
}

#[derive(Serialize, Deserialize)]
pub struct Plate {
    pub name: String,

    pub version: String,

    // [sic!]
    pub field_count: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquisitions: Option<Vec<Acquisition>>,

    pub columns: Vec<Column>,

    pub rows: Vec<Row>,

    pub wells: Vec<Well>,
}

// todo: plate validation
//  - "plate.columns": each column in physical plate must be defined, even if no wells in the columns are defined
//  - "plate.columns[*].name": must contain only alphanumeric characters
//  - "plate.columns[*].name": must be case-sensitive
//  - "plate.columns[*].name": must not be a duplicate of any other name in "plate.columns"
//  - "plate.rows": each row in physical plate must be defined, even if no wells in the rows are defined
//  - "plate.rows[*].name": must contain only alphanumeric characters
//  - "plate.rows[*].name": must be case-sensitive
//  - "plate.rows[*].name": must not be a duplicate of any other name in "plate.columns"
//  - "plate.wells[*].path": must consist of a "plate.rows[*].name", a file separator (/), and a "plate.columns[*].name"
//  - "plate.wells[*].path": must not not contain additional leading or trailing directories
//  - "plate.wells[*].rowIndex": must be an index into "plate.rows[*]"
//  - "plate.wells[*].columnIndex": must be an index into "plate.columns[*]"
//  - "plate.wells": "path" and "rowIndex"+"columnIndex" pair must refer to same row/column pair
