use serde::{Serialize, Deserialize};

// https://ngff.openmicroscopy.org/0.4/#axes-md



#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
pub struct SpaceAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<SpaceUnit>
}

#[derive(Serialize, Deserialize)]
pub struct TimeAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<TimeUnit>,
}

#[derive(Serialize, Deserialize)]
pub struct ChannelAxis {
    pub name: String,
}

#[derive(Serialize, Deserialize)]
pub struct CustomAxis {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub axis_type: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisType {
    Space,
    Time,
    Channel,
    // todo: translate to unknown
    Unknown,
}

// TODO: maybe do this with an enum variant?
/// Represen
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisUnit {
    // Space Units
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

    // Time units
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

impl AxisUnit {
    /// Checks whether the `AxisUnit` is a 'space' axis unit.
    /// See also: `is_time_unit`
    pub fn is_space_unit(&self) -> bool {
        match self {
            AxisUnit::Angstrom |
            AxisUnit::Attometer |
            AxisUnit::Centimeter |
            AxisUnit::Decimeter |
            AxisUnit::Exameter |
            AxisUnit::Femtometer |
            AxisUnit::Foot |
            AxisUnit::Gigameter |
            AxisUnit::Hectometer |
            AxisUnit::Inch |
            AxisUnit::Kilometer |
            AxisUnit::Megameter |
            AxisUnit::Meter |
            AxisUnit::Micrometer |
            AxisUnit::Mile |
            AxisUnit::Millimeter |
            AxisUnit::Anometer |
            AxisUnit::Parsec |
            AxisUnit::Petameter |
            AxisUnit::Picometer |
            AxisUnit::Terameter |
            AxisUnit::Yard |
            AxisUnit::Yoctometer |
            AxisUnit::Yottameter |
            AxisUnit::Zeptometer |
            AxisUnit::Zettameter => true,
            _ => false
        }
    }

    /// Checks whether the `AxisUnit` is a 'time' axis unit.
    /// See also: `is_space_unit`
    pub fn is_time_unit(&self) -> bool {
        !self.is_space_unit()
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


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
