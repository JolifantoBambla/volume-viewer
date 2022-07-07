use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum DataType {
    #[serde(rename = "|u1")]
    Uint8,

    #[serde(rename = "|i1")]
    Int8,

    // todo: also allow 'B'
    #[serde(rename = "|b")]
    Boolean,

    #[serde(rename = "<u1")]
    Uint8LittleEndian,

    #[serde(rename = "<i1")]
    Int8LittleEndian,

    // todo: also allow '<B'
    #[serde(rename = "<b")]
    BooleanLittleEndian,

    #[serde(rename = "<u2")]
    Uint16LittleEndian,

    #[serde(rename = "<i2")]
    Int16LittleEndian,

    #[serde(rename = "<u4")]
    Uint32LittleEndian,

    #[serde(rename = "<i4")]
    Int32LittleEndian,

    #[serde(rename = "<f4")]
    Float32LittleEndian,

    #[serde(rename = "<f8")]
    Float64LittleEndian,

    #[serde(rename = ">u1")]
    Uint8BigEndian,

    #[serde(rename = ">i1")]
    Int8BigEndian,

    // todo: also allow '>B'
    #[serde(rename = ">b")]
    BooleanBigEndian,

    #[serde(rename = ">u2")]
    Uint16BigEndian,

    #[serde(rename = ">i2")]
    Int16BigEndian,

    #[serde(rename = ">u4")]
    Uint32BigEndian,

    #[serde(rename = ">i4")]
    Int32BigEndian,

    #[serde(rename = ">f4")]
    Float32BigEndian,

    #[serde(rename = ">f8")]
    Float64BigEndian,
}
