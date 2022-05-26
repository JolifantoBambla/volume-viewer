use serde::{Serialize, Deserialize};

pub mod v0_1;
pub mod v0_2 {
    pub use crate::omero::v0_1::*;
}
pub mod v0_3 {
    pub use crate::omero::v0_1::*;
}
pub mod v0_4 {
    pub use crate::omero::v0_1::*;
}

// Note: "omero.version" is not explicitly optional, so I treat it as not optional, i.e. it doesn't
// need util::versioned

#[derive(Serialize, Deserialize)]
#[serde(tag = "version")]
pub enum Omero {
    #[serde(rename = "0.4")]
    V0_4(v0_4::Omero),

    #[serde(rename = "0.3")]
    V0_3(v0_3::Omero),

    #[serde(rename = "0.2")]
    V0_2(v0_2::Omero),

    #[serde(rename = "0.1")]
    V0_1(v0_1::Omero),
}
