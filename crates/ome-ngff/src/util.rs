pub(crate) use log::warn;

// todo: add example
/// Checks whether a given boolean expression is true or false, logs a warning if it is false, and returns the result of the boolean expression.
macro_rules! warn_unless {
    ( $condition:expr, $message:expr $(, $format_arg:expr)* $(,)? ) => {
        if $condition {
            true
        } else {
            warn!($message $(,$format_arg)*);
            false
        }
    }
}

pub(crate) use warn_unless;
