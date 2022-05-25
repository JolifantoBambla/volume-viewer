use serde::{Serialize, Deserialize};

pub mod v0_2;
pub mod v0_3;
pub mod v0_4;

#[derive(Serialize, Deserialize)]
#[serde(tag = "version")]
enum TaggedMultiscale {
    #[serde(rename = "0.2")]
    V0_2(v0_2::Multiscale),

    #[serde(rename = "0.3")]
    V0_3(v0_3::Multiscale),

    #[serde(rename = "0.4")]
    V0_4(v0_4::Multiscale),
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum UntaggedMultiscale {
    // Note: the order is important here (newest to oldest) due to the structural similarity between
    // the different versions. E.g. v0.4 is a superset of v0.2, so every v0.4 can be deserialized
    // into a v0.2.
    V0_4(v0_4::Multiscale),
    V0_3(v0_3::Multiscale),
    V0_2(v0_2::Multiscale),
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
    #[serde(rename = "0.2")]
    V0_2(v0_2::Multiscale),

    #[serde(rename = "0.3")]
    V0_3(v0_3::Multiscale),

    #[serde(rename = "0.4")]
    V0_4(v0_4::Multiscale),
}

impl From<MaybeTaggedMultiscale> for Multiscale {
    fn from(multiscale: MaybeTaggedMultiscale) -> Multiscale {
        match multiscale {
            MaybeTaggedMultiscale::Tagged(TaggedMultiscale::V0_4 (m))
            | MaybeTaggedMultiscale::Untagged(UntaggedMultiscale::V0_4(m))
            => Multiscale::V0_4(m),
            MaybeTaggedMultiscale::Tagged(TaggedMultiscale::V0_3 (m))
            | MaybeTaggedMultiscale::Untagged(UntaggedMultiscale::V0_3(m))
            => Multiscale::V0_3(m),
            MaybeTaggedMultiscale::Tagged(TaggedMultiscale::V0_2 (m))
            | MaybeTaggedMultiscale::Untagged(UntaggedMultiscale::V0_2(m))
            => Multiscale::V0_2(m),
        }
    }
}
