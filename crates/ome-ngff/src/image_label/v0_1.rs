use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Color {
    #[serde(rename = "label-value")]
    pub label_value: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
}

#[derive(Serialize, Deserialize)]
pub struct Source {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

/// overlapping labels may be represented by using a specially assigned value, e.g. the highest integer available in the pixel range.
#[derive(Serialize, Deserialize)]
pub struct ImageLabel {
    // if contains duplicate `label_value`s ignore all but the last entry
    pub colors: Vec<Color>,

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
