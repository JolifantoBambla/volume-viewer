use serde::{Serialize, Deserialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Identity {}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Translation {
    Translation(Vec<f32>),
    Path(String),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scale {
    Scale(Vec<f32>),
    Path(String),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(tag = "type")]
pub enum CoordinateTransformation {
    Identity(Identity),
    Translation(Translation),
    Scale(Scale),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{from_str, to_string, Value};

    fn identity_json() -> String {
        r#"{"type": "identity"}"#.to_string()
    }

    fn translation_vec_json() -> String {
        r#"{"type": "translation", "translation": [1.0, 2.0, 3.0]}"#.to_string()
    }

    fn translation_path_json() -> String {
        r#"{"type": "translation", "path": "/path/to/translation"}"#.to_string()
    }

    fn translation_invalid_json() -> String {
        r#"{"type": "translation"}"#.to_string()
    }

    fn scale_vec_json() -> String {
        r#"{"type": "scale", "scale": [1.0, 2.0, 3.0]}"#.to_string()
    }

    fn scale_path_json() -> String {
        r#"{"type": "scale", "path": "/path/to/scale"}"#.to_string()
    }

    fn scale_invalid_json() -> String {
        r#"{"type": "scale"}"#.to_string()
    }

    fn invalid_coordinate_transformation_json() -> String {
        r#"{"translation": [1.0, 2.0, 3.0]}"#.to_string()
    }

    fn identity() -> CoordinateTransformation {
        CoordinateTransformation::Identity(Identity{})
    }

    fn translation_vec() -> CoordinateTransformation {
        CoordinateTransformation::Translation(Translation::Translation(vec![1.0, 2.0, 3.0]))
    }

    fn translation_path() -> CoordinateTransformation {
        CoordinateTransformation::Translation(Translation::Path("/path/to/translation".to_string()))
    }

    fn scale_vec() -> CoordinateTransformation {
        CoordinateTransformation::Scale(Scale::Scale(vec![1.0, 2.0, 3.0]))
    }

    fn scale_path() -> CoordinateTransformation {
        CoordinateTransformation::Scale(Scale::Path("/path/to/scale".to_string()))
    }

    fn coordinate_transformations_json() -> String {
        format!(
            "[{},{},{},{},{}]",
            identity_json(),
            translation_vec_json(),
            translation_path_json(),
            scale_vec_json(),
            scale_path_json(),
        )
    }

    fn coordinate_transformations() -> Vec<CoordinateTransformation> {
        vec![
            identity(),
            translation_vec(),
            translation_path(),
            scale_vec(),
            scale_path(),
        ]
    }

    #[test]
    fn serialize_identity() {
        assert_eq!(
            from_str::<Value>(&to_string(&identity()).unwrap()).unwrap(),
            from_str::<Value>(&identity_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_identity() {
        assert_eq!(
            from_str::<CoordinateTransformation>(&identity_json()).unwrap(),
            identity()
        );
    }

    #[test]
    fn serialize_translation_vec() {
        assert_eq!(
            from_str::<Value>(&to_string(&translation_vec()).unwrap()).unwrap(),
            from_str::<Value>(&translation_vec_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_translation_vec() {
        assert_eq!(
            from_str::<CoordinateTransformation>(&translation_vec_json()).unwrap(),
            translation_vec()
        );
    }

    #[test]
    fn serialize_translation_path() {
        assert_eq!(
            from_str::<Value>(&to_string(&translation_path()).unwrap()).unwrap(),
            from_str::<Value>(&translation_path_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_translation_path() {
        assert_eq!(
            from_str::<CoordinateTransformation>(&translation_path_json()).unwrap(),
            translation_path()
        );
    }

    #[test]
    fn serialize_scale_vec() {
        assert_eq!(
            from_str::<Value>(&to_string(&scale_vec()).unwrap()).unwrap(),
            from_str::<Value>(&scale_vec_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_scale_vec() {
        assert_eq!(
            from_str::<CoordinateTransformation>(&scale_vec_json()).unwrap(),
            scale_vec()
        );
    }

    #[test]
    fn serialize_scale_path() {
        assert_eq!(
            from_str::<Value>(&to_string(&scale_path()).unwrap()).unwrap(),
            from_str::<Value>(&scale_path_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_scale_path() {
        assert_eq!(
            from_str::<CoordinateTransformation>(&scale_path_json()).unwrap(),
            scale_path()
        );
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_translation() {
        from_str::<CoordinateTransformation>(&translation_invalid_json()).unwrap();
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_scale() {
        from_str::<CoordinateTransformation>(&scale_invalid_json()).unwrap();
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_coordinate_transformation() {
        from_str::<CoordinateTransformation>(&invalid_coordinate_transformation_json()).unwrap();
    }

    #[test]
    fn serialize_coordinate_transformations() {
        assert_eq!(
            from_str::<Value>(&to_string(&coordinate_transformations()).unwrap()).unwrap(),
            from_str::<Value>(&coordinate_transformations_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_coordinate_transformations() {
        assert_eq!(
            from_str::<Vec<CoordinateTransformation>>(&coordinate_transformations_json()).unwrap(),
            coordinate_transformations()
        );
    }
}
