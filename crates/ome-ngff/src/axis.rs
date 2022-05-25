use serde::{Serialize, Deserialize, Serializer};

// https://ngff.openmicroscopy.org/0.4/#axes-md

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpaceUnit {
    Angstrom,
    Attometer,
    Centimeter,
    Decimeter,
    Exameter,
    Femtometer,
    Foot,
    Gigameter,
    Hectometer,
    Inch,
    Kilometer,
    Megameter,
    Meter,
    Micrometer,
    Mile,
    Millimeter,
    Anometer,
    Parsec,
    Petameter,
    Picometer,
    Terameter,
    Yard,
    Yoctometer,
    Yottameter,
    Zeptometer,
    Zettameter,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeUnit {
    Attosecond,
    Centisecond,
    Day,
    Decisecond,
    Exasecond,
    Femtosecond,
    Gigasecond,
    Hectosecond,
    Hour,
    Kilosecond,
    Megasecond,
    Microsecond,
    Millisecond,
    Minute,
    Nanosecond,
    Petasecond,
    Picosecond,
    Second,
    Terasecond,
    Yoctosecond,
    Yottasecond,
    Zeptosecond,
    Zettasecond
}

// axis types

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpaceAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<SpaceUnit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<TimeUnit>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelAxis {
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub axis_type: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

// https://github.com/serde-rs/serde/issues/1799#issuecomment-624978919

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
enum TaggedAxis {
    Space(SpaceAxis),
    Time(TimeAxis),
    Channel(ChannelAxis),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MaybeTaggedAxis {
    Tagged(TaggedAxis),
    Untagged(CustomAxis),
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(from = "MaybeTaggedAxis")]
#[serde(tag = "type")]
pub enum Axis {
    Space(SpaceAxis),
    Time(TimeAxis),
    Channel(ChannelAxis),
    Custom(CustomAxis),
}

impl Axis {
    pub fn get_name(&self) -> String {
        match self {
            Axis::Space(axis) => {
                axis.name.to_string()
            },
            Axis::Time(axis) => {
                axis.name.to_string()
            },
            Axis::Channel(axis) => {
                axis.name.to_string()
            },
            Axis::Custom(axis) => {
                axis.name.to_string()
            },
        }
    }

    pub fn clone_as_space_axis(&self) -> Option<SpaceAxis> {
        match self {
            Axis::Space(axis) => Some(axis.clone()),
            _ => None
        }
    }

    pub fn clone_as_time_axis(&self) -> Option<TimeAxis> {
        match self {
            Axis::Time(axis) => Some(axis.clone()),
            _ => None
        }
    }

    pub fn clone_as_channel_axis(&self) -> Option<ChannelAxis> {
        match self {
            Axis::Channel(axis) => Some(axis.clone()),
            _ => None
        }
    }

    pub fn clone_as_custom_axis(&self) -> Option<CustomAxis> {
        match self {
            Axis::Custom(axis) => Some(axis.clone()),
            _ => None
        }
    }
}

// custom Serialize implementation because otherwise CustomAxis instance without an axis_type get a "type": "custom"
impl Serialize for Axis {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        match self {
            Axis::Custom(value) => {
                self::CustomAxis::serialize(value, serializer)
            },
            Axis::Space(value) => {
                TaggedAxis::serialize(&TaggedAxis::Space(value.clone()), serializer)
            }
            Axis::Time(value) => {
                TaggedAxis::serialize(&TaggedAxis::Time(value.clone()), serializer)
            }
            Axis::Channel(value) => {
                TaggedAxis::serialize(&TaggedAxis::Channel(value.clone()), serializer)
            }
        }
    }
}

impl From<MaybeTaggedAxis> for Axis {
    fn from(axis: MaybeTaggedAxis) -> Axis {
        match axis {
            MaybeTaggedAxis::Untagged(CustomAxis{ name, axis_type, unit})
            => Axis::Custom(CustomAxis { name, axis_type, unit }),
            MaybeTaggedAxis::Tagged(TaggedAxis::Space(SpaceAxis{ name, unit }))
            => Axis::Space(SpaceAxis { name, unit }),
            MaybeTaggedAxis::Tagged(TaggedAxis::Time(TimeAxis{ name, unit }))
            => Axis::Time(TimeAxis { name, unit }),
            MaybeTaggedAxis::Tagged(TaggedAxis::Channel(ChannelAxis{ name }))
            => Axis::Channel(ChannelAxis { name }),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{from_str, to_string, Value};
    use super::*;

    fn space_no_unit_json() -> String {
        r#"{"name": "x", "type": "space"}"#.to_string()
    }

    fn space_json() -> String {
        r#"{"name": "y", "type": "space", "unit": "millimeter"}"#.to_string()
    }

    fn space_no_unit() -> SpaceAxis {
        SpaceAxis{
            name: "x".to_string(),
            unit: None,
        }
    }

    fn space() -> SpaceAxis {
        SpaceAxis{
            name: "y".to_string(),
            unit: Some(SpaceUnit::Millimeter),
        }
    }

    fn time_no_unit_json() -> String {
        r#"{"name": "t", "type": "time"}"#.to_string()
    }

    fn time_json() -> String {
        r#"{"name": "t", "type": "time", "unit": "millisecond"}"#.to_string()
    }

    fn time_no_unit() -> TimeAxis {
        TimeAxis {
            name: "t".to_string(),
            unit: None,
        }
    }

    fn time() -> TimeAxis {
        TimeAxis{
            name: "t".to_string(),
            unit: Some(TimeUnit::Millisecond),
        }
    }

    fn channel_no_unit_json() -> String {
        r#"{"name": "c", "type": "channel"}"#.to_string()
    }

    fn channel_json() -> String {
        r#"{"name": "c", "type": "channel", "unit": "should get ingnored"}"#.to_string()
    }

    fn channel() -> ChannelAxis {
        ChannelAxis{
            name: "c".to_string(),
        }
    }

    fn custom_no_unit_no_type_json() -> String {
        r#"{"name": "foo"}"#.to_string()
    }

    fn custom_no_unit_json() -> String {
        r#"{"name": "foo", "type": "bar"}"#.to_string()
    }

    fn custom_no_type_json() -> String {
        r#"{"name": "foo", "unit": "bar"}"#.to_string()
    }

    fn custom_json() -> String {
        r#"{"name": "foo", "type": "bar", "unit": "baz"}"#.to_string()
    }

    fn custom_no_unit_no_type() -> CustomAxis {
        CustomAxis{
            name: "foo".to_string(),
            axis_type: None,
            unit: None,
        }
    }

    fn custom_no_unit() -> CustomAxis {
        CustomAxis{
            name: "foo".to_string(),
            axis_type: Some("bar".to_string()),
            unit: None,
        }
    }

    fn custom_no_type() -> CustomAxis {
        CustomAxis{
            name: "foo".to_string(),
            axis_type: None,
            unit: Some("bar".to_string()),
        }
    }

    fn custom() -> CustomAxis {
        CustomAxis{
            name: "foo".to_string(),
            axis_type: Some("bar".to_string()),
            unit: Some("baz".to_string()),
        }
    }

    fn space_invalid_json() -> String {
        r#"{"type": "space"}"#.to_string()
    }

    fn time_invalid_json() -> String {
        r#"{"type": "time"}"#.to_string()
    }

    fn channel_invalid_json() -> String {
        r#"{"type": "channel"}"#.to_string()
    }

    fn custom_invalid_json() -> String {
        r#"{"type": "foo"}"#.to_string()
    }

    fn axes_json(exclude_channel_unit: bool) -> String {
        format!(
            "[{},{},{},{},{},{},{},{},{},{}]",
            space_no_unit_json(),
            space_json(),
            time_no_unit_json(),
            time_json(),
            channel_no_unit_json(),
            if exclude_channel_unit { channel_no_unit_json() } else { channel_json() },
            custom_no_unit_no_type_json(),
            custom_no_unit_json(),
            custom_no_type_json(),
            custom_json()
        )
    }

    fn axes() -> Vec<Axis> {
        vec![
            Axis::Space(space_no_unit()),
            Axis::Space(space()),
            Axis::Time(time_no_unit()),
            Axis::Time(time()),
            Axis::Channel(channel()),
            Axis::Channel(channel()),
            Axis::Custom(custom_no_unit_no_type()),
            Axis::Custom(custom_no_unit()),
            Axis::Custom(custom_no_type()),
            Axis::Custom(custom()),
        ]
    }

    #[test]
    fn serialize_space_without_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&Axis::Space(space_no_unit())).unwrap()).unwrap(),
            from_str::<Value>(&space_no_unit_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_space_without_unit() {
        assert_eq!(
            from_str::<SpaceAxis>(&space_no_unit_json()).unwrap(),
            space_no_unit()
        );
    }

    #[test]
    fn serialize_space_with_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&&Axis::Space(space())).unwrap()).unwrap(),
            from_str::<Value>(&space_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_space_with_unit() {
        assert_eq!(
            from_str::<SpaceAxis>(&space_json()).unwrap(),
            space()
        );
    }

    #[test]
    fn serialize_time_without_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&Axis::Time(time_no_unit())).unwrap()).unwrap(),
            from_str::<Value>(&time_no_unit_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_time_without_unit() {
        assert_eq!(
            from_str::<TimeAxis>(&time_no_unit_json()).unwrap(),
            time_no_unit()
        );
    }

    #[test]
    fn serialize_time_with_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&Axis::Time(time())).unwrap()).unwrap(),
            from_str::<Value>(&time_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_time_with_unit() {
        assert_eq!(
            from_str::<TimeAxis>(&time_json()).unwrap(),
            time()
        );
    }

    #[test]
    fn serialize_channel_without_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&Axis::Channel(channel())).unwrap()).unwrap(),
            from_str::<Value>(&channel_no_unit_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_channel_without_unit() {
        assert_eq!(
            from_str::<ChannelAxis>(&channel_no_unit_json()).unwrap(),
            channel()
        );
    }

    #[test]
    fn deserialize_channel_with_unit() {
        assert_eq!(
            from_str::<ChannelAxis>(&channel_json()).unwrap(),
            channel()
        );
    }

    #[test]
    fn serialize_custom_without_type_without_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&Axis::Custom(custom_no_unit_no_type())).unwrap()).unwrap(),
            from_str::<Value>(&custom_no_unit_no_type_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_custom_without_type_without_unit() {
        assert_eq!(
            from_str::<CustomAxis>(&custom_no_unit_no_type_json()).unwrap(),
            custom_no_unit_no_type()
        );
    }

    #[test]
    fn serialize_custom_without_type_with_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&Axis::Custom(custom_no_type())).unwrap()).unwrap(),
            from_str::<Value>(&custom_no_type_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_custom_without_type_with_unit() {
        assert_eq!(
            from_str::<CustomAxis>(&custom_no_type_json()).unwrap(),
            custom_no_type()
        );
    }

    #[test]
    fn serialize_custom_with_type_with_unit() {
        assert_eq!(
            from_str::<Value>(&to_string(&Axis::Custom(custom_no_unit())).unwrap()).unwrap(),
            from_str::<Value>(&custom_no_unit_json()).unwrap()
        );
    }

    #[test]
    fn deserialize_custom_with_type_with_unit() {
        assert_eq!(
            from_str::<CustomAxis>(&custom_no_unit_json()).unwrap(),
            custom_no_unit()
        );
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_space() {
        from_str::<Axis>(&space_invalid_json()).unwrap();
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_time() {
        from_str::<Axis>(&time_invalid_json()).unwrap();
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_channel() {
        from_str::<Axis>(&channel_invalid_json()).unwrap();
    }

    #[test]
    #[should_panic]
    fn deserialize_invalid_custom() {
        from_str::<Axis>(&custom_invalid_json()).unwrap();
    }

    #[test]
    fn serialize_axes() {
        assert_eq!(
            from_str::<Value>(&to_string(&axes()).unwrap()).unwrap(),
            from_str::<Value>(&axes_json(true)).unwrap()
        );
    }

    #[test]
    fn deserialize_axes() {
        assert_eq!(
            from_str::<Vec<Axis>>(&axes_json(false)).unwrap(),
            axes()
        );
    }
}
