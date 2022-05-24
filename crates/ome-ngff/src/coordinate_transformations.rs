use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Identity {}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Translation {
    Translation(Vec<f32>),
    Path(String),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scale {
    Scale(Vec<f32>),
    Path(String),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "type")]
pub enum CoordinateTransformation {
    Identity(Identity),
    Translation(Translation),
    Scale(Scale),
}

#[cfg(test)]
mod tests {
    // todo:
    //  - identity with extra fields
    //  - identity without extra fields
    //  - translation with vec
    //  - translation with path
    //  - scale with vec
    //  - scale with path
    //  - coordinate transformation without "type" should throw

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
