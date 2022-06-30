pub mod zarr_v2 {
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::collections::HashMap;

    #[derive(Serialize, Deserialize)]
    pub struct Compressor {
        pub id: String,

        #[serde(flatten)]
        pub meta_data: HashMap<String, Value>,
    }

    #[derive(Serialize, Deserialize)]
    pub enum Order {
        /// “C” means row-major order, i.e., the last dimension varies fastest.
        C,

        /// “F” means column-major order, i.e., the first dimension varies fastest.
        F,
    }

    #[derive(Serialize, Deserialize)]
    pub enum DimensionSeparator {
        #[serde(rename = ".")]
        Dot,

        #[serde(rename = "/")]
        Slash,
    }

    impl Default for DimensionSeparator {
        fn default() -> Self {
            DimensionSeparator::Dot
        }
    }

    #[derive(Serialize, Deserialize)]
    pub struct Filter {
        pub id: String,

        #[serde(flatten)]
        pub meta_data: HashMap<String, Value>,
    }

    /// Each array requires essential configuration metadata to be stored, enabling correct
    /// interpretation of the stored data. This metadata is encoded using JSON and stored as the
    /// value of the “.zarray” key within an array store.
    #[derive(Serialize, Deserialize)]
    pub struct ArrayMetadata {
        /// An integer defining the version of the storage specification to which the array store
        /// adheres.
        pub zarr_format: u32,

        /// A list of integers defining the length of each dimension of the array.
        pub shape: Vec<usize>,

        /// A list of integers defining the length of each dimension of a chunk of the array. Note
        /// that all chunks within a Zarr array have the same shape.
        pub chunks: Vec<usize>,

        // todo: string or list - needs to be parsed to a rust repesentation
        /// A string or list defining a valid data type for the array. See also the subsection below on data type encoding.
        #[serde(rename = "dtype")]
        pub data_type: String,

        // todo: anything we know about compressors?
        /// A JSON object identifying the primary compression codec and providing configuration
        /// parameters, or null if no compressor is to be used. The object MUST contain an "id" key
        /// identifying the codec to be used.
        pub compressor: Option<Compressor>,

        /// A scalar value providing the default value to use for uninitialized portions of the
        /// array, or null if no fill_value is to be used.
        pub fill_value: Option<Value>,

        /// Either “C” or “F”, defining the layout of bytes within each chunk of the array. “C”
        /// means row-major order, i.e., the last dimension varies fastest; “F” means column-major
        /// order, i.e., the first dimension varies fastest.
        pub order: Order,

        /// A list of JSON objects providing codec configurations, or null if no filters are to be
        /// applied. Each codec configuration object MUST contain a "id" key identifying the codec
        /// to be used.
        pub filters: Option<Vec<Filter>>,

        /// If present, either the string "." or "/"" defining the separator placed between the
        /// dimensions of a chunk. If the value is not set, then the default MUST be assumed to be
        /// ".", leading to chunk keys of the form “0.0”. Arrays defined with "/" as the dimension
        /// separator can be considered to have nested, or hierarchical, keys of the form “0/0” that
        /// SHOULD where possible produce a directory-like structure.
        #[serde(default)]
        pub dimension_separator: DimensionSeparator,
        // Other keys SHOULD NOT be present within the metadata object and SHOULD be ignored by
        // implementations.
    }

    // todo: fill_value validation & usage: https://zarr.readthedocs.io/en/stable/spec/v2.html#fill-value-encoding
}

type OMEZarrAttributes = ome_ngff::metadata::Metadata;

fn main() {
    println!("Hello, world!");
}
