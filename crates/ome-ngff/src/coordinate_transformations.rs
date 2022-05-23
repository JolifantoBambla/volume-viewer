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
pub enum CoordinateTransformation {
    Identity,
    Translation,
    Scale,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
