use serde::{Serialize, Deserialize};

pub mod v0_1 {
    use serde::{Serialize, Deserialize};

    #[derive(Serialize, Deserialize)]
    pub struct Image {
        pub path: String,

        #[serde(skip_serializing_if = "Option::is_none")]
        pub acquisition: Option<u64>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Well {
        pub images: Vec<Image>,
    }
}
pub mod v0_2 {
    pub use crate::well::v0_1::*;
}
pub mod v0_3 {
    pub use crate::well::v0_1::*;
}
pub mod v0_4 {
    pub use crate::well::v0_1::*;
}

// Note: "well.version" is not explicitly optional, so I treat it as not optional, i.e. it doesn't
// need util::versioned

#[derive(Serialize, Deserialize)]
#[serde(tag = "version")]
pub enum Well {
    #[serde(rename = "0.4")]
    V0_4(v0_4::Well),

    #[serde(rename = "0.3")]
    V0_3(v0_3::Well),

    #[serde(rename = "0.2")]
    V0_2(v0_2::Well),

    #[serde(rename = "0.1")]
    V0_1(v0_1::Well),
}