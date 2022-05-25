use serde::{Serialize, Deserialize};

pub mod v0_3;
pub mod v0_4;

#[derive(Serialize, Deserialize)]
#[serde(tag = "version")]
enum TaggedMultiscale {
    #[serde(rename = "0.3")]
    V0_3(v0_3::Multiscale),

    #[serde(rename = "0.4")]
    V0_4(v0_4::Multiscale),
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum UntaggedMultiscale {
    V0_3(v0_3::Multiscale),
    V0_4(v0_4::Multiscale),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MaybeTaggedMultiscale {
    Tagged(TaggedMultiscale),
    Untagged(UntaggedMultiscale),
}

#[derive(Serialize, Deserialize)]
#[serde(from = "MaybeTaggedMultiscale")]
#[serde(tag = "version")]
pub enum Multiscale {
    #[serde(rename = "0.3")]
    V0_3(v0_3::Multiscale),

    #[serde(rename = "0.4")]
    V0_4(v0_4::Multiscale),
}

impl From<MaybeTaggedMultiscale> for Multiscale {
    fn from(multiscale: MaybeTaggedMultiscale) -> Multiscale {
        match multiscale {
            MaybeTaggedMultiscale::Tagged(TaggedMultiscale::V0_3 (m))
            | MaybeTaggedMultiscale::Untagged(UntaggedMultiscale::V0_3(m))
            => Multiscale::V0_3(m),
            MaybeTaggedMultiscale::Tagged(TaggedMultiscale::V0_4 (m))
            | MaybeTaggedMultiscale::Untagged(UntaggedMultiscale::V0_4(m))
            => Multiscale::V0_4(m),
        }
    }
}
