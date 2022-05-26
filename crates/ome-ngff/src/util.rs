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

macro_rules! versioned {
    ($name:ident { $($field_name:ident($variant:ty : $version_name:expr)),+ $(,)? }) => {
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