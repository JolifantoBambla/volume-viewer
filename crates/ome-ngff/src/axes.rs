use std::fmt::{format, Formatter};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::{MapAccess, Visitor};

// https://ngff.openmicroscopy.org/0.4/#axes-md



#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct SpaceAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<SpaceUnit>,
}

impl SpaceAxis {
    pub fn new(name: String, unit: Option<SpaceUnit>) -> Self {
        Self{
            name,
            unit,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TimeAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<TimeUnit>,
}

impl TimeAxis {
    pub fn new(name: String, unit: Option<TimeUnit>) -> Self {
        Self{
            name,
            unit,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChannelAxis {
    pub name: String,
}

impl ChannelAxis {
    pub fn new(name: String) -> Self {
        Self{
            name,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CustomAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub axis_type: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

impl CustomAxis {
    pub fn new(name: String, axis_type: Option<String>, unit: Option<String>) -> Self {
        Self{
            name,
            axis_type,
            unit,
        }
    }
}

// https://github.com/serde-rs/serde/issues/1799#issuecomment-624978919

#[derive(Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(from = "MaybeTaggedAxis")]
#[serde(tag = "type")]
pub enum Axis {
    Space(SpaceAxis),
    Time(TimeAxis),
    Channel(ChannelAxis),
    Custom(CustomAxis),
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

/**
impl Serialize for Axis {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        match self {
            Axis::Space(value) => {
                self::SpaceAxis::serialize(value, serializer)
            }
            Axis::Time(value) => {
                self::TimeAxis::serialize(value, serializer)
            }
            Axis::Channel(value) => {
                self::ChannelAxis::serialize(value, serializer)
            }
            Axis::Custom(value) => {
                self::CustomAxis::serialize(value, serializer)
            }
        }
    }
}


impl<'de>  Deserialize<'de> for Axis {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        struct AxisVisitor;
        impl<'de> Visitor<'de> for AxisVisitor {
            type Value = Axis;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str("an Axis variant")
            }

            fn visit_map<V>(self, mut access: V) -> Result<Self::Value, V::Error> where V: MapAccess<'de> {
                let mut name = "".to_string();
                let mut axis_type: Option<String> = None;
                let mut unit: Option<String> = None;
                while let Some((key, value)) = access.next_entry()? {
                    match key {
                        "name" => {
                            name = value;
                        },
                        "type" => {
                            axis_type = Some(value);
                        },
                        "unit" => {
                            unit = Some(value);
                        },
                        // ignore other keys
                        _ => {}
                    }
                }

                // todo: err on empty name

                if axis_type.is_some() {
                    match axis_type.unwrap().as_str() {
                        "space" => {
                            // todo: err on invalid unit
                            Ok(Axis::Space(SpaceAxis::new(
                                name.to_string(),
                                None
                            )))
                        }
                        "time" => {
                            // todo: err on invalid unit
                            Ok(Axis::Time(TimeAxis::new(
                                name.to_string(),
                                None
                            )))
                        },
                        "channel" => {
                            // todo: warn on lost unit
                            Ok(Axis::Channel(ChannelAxis::new(
                                name.to_string()
                            )))
                        },
                        _ => {
                            Ok(Axis::Custom(CustomAxis::new(
                                name.to_string(),
                                axis_type,
                                unit
                            )))
                        }
                    }
                } else {
                    Ok(Axis::Custom(CustomAxis::new(
                        name,
                        axis_type,
                        unit
                    )))
                }
            }
        }
        deserializer.deserialize_any(AxisVisitor)
    }
}
*/

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
