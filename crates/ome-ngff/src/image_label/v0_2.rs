use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value;

pub use crate::image_label::v0_1::{Color, Source};

#[derive(Serialize, Deserialize)]
pub struct Property {
    #[serde(rename = "label-value")]
    pub label_value: u64,

    #[serde(rename = "area (pixels)")]
    pub area: u64,

    // an arbitrary number of key-value pairs may be present for each label
    // not all label values must share the same key value-pairs within the properties list
    #[serde(flatten)]
    pub extra: HashMap<String, Value>
}

/// overlapping labels may be represented by using a specially assigned value, e.g. the highest integer available in the pixel range.
#[derive(Serialize, Deserialize)]
pub struct ImageLabel {
    // if contains duplicate `label_value`s ignore all but the last entry
    pub colors: Vec<Color>,

    pub properties: Vec<Property>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
}

impl ImageLabel {
    pub fn is_valid(&self) -> bool {
        true
    }

// todo: image-label
//  - "image-label.colors": "label-value"s should be unique
}

#[cfg(test)]
mod tests {
    //use super::*;

    // todo: the example in the spec is an invalid JSON, see https://github.com/ome/ngff/issues/125
    //fn spec_example_json() -> String {
    //    r#""#.to_string()
    //}
}
