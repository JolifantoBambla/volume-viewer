use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::path::Display;
use std::str::FromStr;
use std::string::ParseError;

use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::{EnumAccess, Error, MapAccess, SeqAccess, Visitor};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisType {
    Space,
    Time,
    Channel,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisSpaceUnit {
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

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisTimeUnit {
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

#[serde(rename_all = "lowercase")]
pub enum AxisUnit {
    AxisSpaceUnit(AxisSpaceUnit),
    AxisTimeUnit(AxisTimeUnit),
}

impl Serialize for AxisUnit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        match self {
            AxisUnit::AxisSpaceUnit(value) => {
                self::AxisSpaceUnit::serialize(value, serializer)
            }
            AxisUnit::AxisTimeUnit(value) => {
                self::AxisTimeUnit::serialize(value, serializer)
            },
        }
    }
}

impl FromStr for AxisUnit {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "angstrom" |
            "attometer" |
            "centimeter" |
            "decimeter" |
            "exameter" |
            "femtometer" |
            "foot" |
            "gigameter" |
            "hectometer" |
            "inch" |
            "kilometer" |
            "megameter" |
            "meter" |
            "micrometer" |
            "mile" |
            "millimeter" |
            "anometer" |
            "parsec" |
            "petameter" |
            "picometer" |
            "terameter" |
            "yard" |
            "yoctometer" |
            "yottameter" |
            "zeptometer" |
            "zettameter" => {Ok(Self::AxisSpaceUnit(AxisSpaceUnit::Angstrom))},
            "Attosecond" |
            "Centisecond" |
            "Day" |
            "Decisecond" |
            "Exasecond" |
            "Femtosecond" |
            "Gigasecond" |
            "Hectosecond" |
            "Hour" |
            "Kilosecond" |
            "Megasecond" |
            "Microsecond" |
            "Millisecond" |
            "Minute" |
            "Nanosecond" |
            "Petasecond" |
            "Picosecond" |
            "Second" |
            "Terasecond" |
            "Yoctosecond" |
            "Yottasecond" |
            "Zeptosecond" |
            "Zettasecond" => {Ok(Self::AxisTimeUnit(AxisTimeUnit::Attosecond))},
            _ => {
                let err_msg = format!("Can't parse AxisUnit from {}", s);
                Err(ParseError::from(()))
            }
        }
    }
}

// use this example to deserialze to one of the two axis unit types: https://serde.rs/string-or-struct.html
impl<'de> Deserialize<'de> for AxisUnit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        struct AxisUnitDeserializer;
        impl<'de> Visitor<'de> for AxisUnitDeserializer {
            type Value = AxisUnit;
            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("AxisUnit")
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> where E: Error {
                AxisUnit::from_str(v)
            }
        }
        deserializer.deserialize_any(AxisUnitDeserializer)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Axis {
    pub name: String,
    // todo: this should be "if is Unknown"

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub axis_type: Option<AxisType>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<AxisUnit>,
}

// https://ngff.openmicroscopy.org/latest/#metadata

#[derive(Serialize, Deserialize)]
pub struct Metadata {

}

#[derive(Serialize, Deserialize)]
pub struct Example {
    pub field1: HashMap<u32, String>,
    pub field2: Vec<Vec<f32>>,
    pub field3: [f32; 4],
    pub axis: Axis,
}

pub fn foo() {
    log::info!("logging from ome-ngff");
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
