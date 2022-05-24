use serde::{Serialize, Deserialize};

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

#[cfg(test)]
mod tests {
    // todo:
    //  - space without unit
    //  - space with unit
    //  - time without unit
    //  - time with unit
    //  - channel without unit
    //  - channel with unit (should be same as without)
    //  - custom axis without unit and type
    //  - custom axis with unit and without type
    //  - custom axis without unit and with type
    //  - axis without name should throw

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
