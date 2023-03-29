use std::fmt;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, Clone)]
pub struct IncludeError {
    line_number: usize,
    identifier: String,
    file: String,
}

impl IncludeError {
    pub fn new(line_number: usize, identifier: String, file: String) -> Self {
        Self {
            line_number,
            identifier,
            file,
        }
    }
}

impl Display for IncludeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}, line {}: could not include \"{}\"",
            self.file, self.line_number, self.identifier
        )
    }
}

#[derive(Debug, Clone)]
pub enum IncludeResolveError {
    NotFound(IncludeError),
    RecursiveInclude(IncludeError),
}

impl Display for IncludeResolveError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IncludeResolveError::NotFound(err) => {
                write!(f, "{err}: no source found for identifier")
            }
            IncludeResolveError::RecursiveInclude(err) => {
                write!(f, "{err}: recursive include")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum PreprocessorError {
    IncludeResolveError(IncludeResolveError),
}

impl Display for PreprocessorError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            PreprocessorError::IncludeResolveError(err) => Display::fmt(&err, f),
        }
    }
}
