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

pub(crate) use warn_unless;
