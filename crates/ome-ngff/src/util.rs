// todo: add example
/// Checks whether a given boolean expression is true or false, logs a warning if it is false, and returns the result of the boolean expression.
#[cfg(feature = "log")]
macro_rules! warn_unless {
    ( $condition:expr, $message:expr $(, $format_arg:expr)* $(,)? ) => {
        if $condition {
            true
        } else {
            use log;
            log::warn!($message $(,$format_arg)*);
            false
        }
    }
}

#[cfg(not(feature = "log"))]
macro_rules! warn_unless {
    ( $condition:expr $(, $_:expr)* $(,)? ) => {
        $condition
    }
}

// todo: proper docs
/// Creates a serde-capable enum for structs that might be versioned by an optional "version" key,
/// but might also only be distinguishable by their structure.
/// See https://ngff.openmicroscopy.org/latest/#multiscale-md
///
/// Note that until `std::concat_idents` (https://doc.rust-lang.org/std/macro.concat_idents.html) is
/// stable only one occurrence per file is valid.
///
/// Note that if a variant is structurally a subset of another the superset must be listed above the
/// subset. (E.g. `multiscale::v0_2::Multiscale` and `multiscale::v0_4::Multiscale`).
macro_rules! versioned {
    ($name:ident { $($field_name:ident($variant:ty : $version_name:expr)),+ $(,)? }) => {
        use serde::{Serialize, Deserialize};

        #[derive(Serialize, Deserialize)]
        #[serde(tag = "version")]
        enum Tagged {
            $(#[serde(rename = $version_name)]
            $field_name($variant)),*
        }

        #[derive(Serialize, Deserialize)]
        #[serde(untagged)]
        enum Untagged {
            // Note: the order is important here (newest to oldest) due to the structural similarity between
            // the different versions. E.g. v0.4 is a superset of v0.2, so every v0.4 can be deserialized
            // into a v0.2.
            $($field_name($variant)),*
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum MaybeTagged {
            Tagged(Tagged),
            Untagged(Untagged),
        }

        #[derive(Serialize, Deserialize)]
        #[serde(from = "MaybeTagged")]
        #[serde(tag = "version")]
        pub enum $name {
            $(#[serde(rename = $version_name)]
            $field_name($variant)),*
        }

        impl From<MaybeTagged> for $name {
            fn from(maybe_tagged: MaybeTagged) -> $name {
                match maybe_tagged {
                    $(MaybeTagged::Tagged(Tagged::$field_name(x))
                    | MaybeTagged::Untagged(Untagged::$field_name(x))
                    => $name::$field_name(x)),*
                }
            }
        }
    }
}

pub(crate) use warn_unless;
pub(crate) use versioned;
