use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum PersistenceMode {
    #[serde(rename = "r")]
    ReadOnly,

    #[serde(rename = "r+")]
    ReadWrite,

    #[serde(rename = "a")]
    ReadWriteNonExisting,

    #[serde(rename = "w")]
    Create,

    #[serde(rename = "w-")]
    CreateNonExisting,
}
