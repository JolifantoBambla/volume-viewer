use std::collections::HashMap;

use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisType {
    Space,
    Time,
    Channel,
    Unknown
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

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisUnit {
    AxisSpaceUnit(AxisSpaceUnit),
    AxisTimeUnit(AxisTimeUnit),
    Unknown,
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
            _ => {
                serializer.serialize_str("")
            }
        }
    }
}

// use this example to deserialze to one of the two axis unit types: https://serde.rs/string-or-struct.html
//impl<'de> Deserialize<'de> for AxisUnit {
//    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {}
//}

#[derive(Serialize, Deserialize)]
pub struct Axis {
    pub name: String,
    pub axis_type: AxisType,
    pub unit: AxisUnit,
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
