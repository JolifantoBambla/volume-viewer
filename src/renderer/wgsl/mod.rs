use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use regex::Regex;

#[derive(Debug, Clone)]
pub struct CouldNotResolve {
    line_number: usize,
    identifier: String,
    file: String,
}

impl Display for CouldNotResolve {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Could not resolve include \"{}\" at {}, line {}", self.identifier, self.file, self.line_number)
    }
}

#[derive(Debug, Clone)]
pub struct RecursiveInclude {
    line_number: usize,
    identifier: String,
    file: String,
}

impl Display for RecursiveInclude {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Could not resolve include \"{}\" at {}, line {}", self.identifier, self.file, self.line_number)
    }
}

#[derive(Debug, Clone)]
pub enum IncludeResolveError {
    ColdNotResolve(CouldNotResolve),
    RecursiveInclude(RecursiveInclude),
}

impl Display for IncludeResolveError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            IncludeResolveError::ColdNotResolve(err) => {
                Display::fmt(&err, f)
            }
            IncludeResolveError::RecursiveInclude(err) => {
                Display::fmt(&err, f)
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
            PreprocessorError::IncludeResolveError(err) => {
                Display::fmt(&err, f)
            }
        }
    }
}

#[derive(Debug)]
pub struct WGSLPreprocessor {
    include_files: HashMap<String, String>,
    include_regex: Regex,
}

impl Default for WGSLPreprocessor {
    fn default() -> Self {
        Self {
            include_files: HashMap::new(),
            include_regex: Regex::new(r"^@include\((?P<identifier>\S+)\)").expect("Could not construct regex"),
        }
    }
}

impl WGSLPreprocessor {
    pub fn new(include_syntax: &str) -> Self {
        WGSLPreprocessor {
            include_regex: Regex::new(include_syntax).expect("Could not construct regex"),
            ..Default::default()
        }
    }

    pub fn include<K, V>(&mut self, identifier: K, source: V) -> &mut Self
        where K: Into<String>, V: Into<String> {
        self.include_files.insert(identifier.into(), source.into());
        self
    }

    pub fn resolve_includes<S>(&self, source: S) -> Result<String, IncludeResolveError>
        where S: Into<String> {
        let src = source.into();
        let mut resolved_includes = HashSet::new();
        let mut include_stack = Vec::new();
        let mut resolved_source = Vec::new();

        let mut source_stack = vec![src.lines().enumerate()];
        while !source_stack.is_empty() {
            let mut lines = source_stack.pop().unwrap();
            let mut resolved = true;
            while let Some((line_number, line)) = lines.next() {
                if let Some(includes) = self.include_regex.captures(line) {
                    let identifier = includes
                        .name("identifier")
                        .expect("Capture group \"identifier\" does not exist.")
                        .as_str();

                    if resolved_includes.contains(identifier) {
                        continue;
                    }

                    if include_stack.contains(&identifier) {
                        return Err(IncludeResolveError::RecursiveInclude(
                            RecursiveInclude {
                                line_number,
                                identifier: identifier.to_string(),
                                file: include_stack.pop().unwrap_or("source").to_string()
                            }
                        ));
                    }

                    if let Some(source) = self.include_files.get(identifier) {
                        include_stack.push(&*identifier);
                        source_stack.push(lines);
                        source_stack.push(source.lines().enumerate());
                        resolved = false;
                        break;
                    } else {
                        return Err(IncludeResolveError::ColdNotResolve(
                            CouldNotResolve {
                                line_number,
                                identifier: identifier.to_string(),
                                file: include_stack.pop().unwrap_or("source").to_string()
                            }
                        ));
                    }
                } else {
                    resolved_source.push(String::from(line));
                }
            }
            if resolved {
                if let Some(resolved_include) = include_stack.pop() {
                    resolved_includes.insert(resolved_include);
                }
            }
        }
        Ok(resolved_source.join("\n"))
    }

    pub fn preprocess<S>(&self, source: S) -> Result<String, PreprocessorError>
        where S: Into<String> {
        let resolved_includes = self.resolve_includes(source);
        if resolved_includes.is_err() {
            return Err(PreprocessorError::IncludeResolveError(resolved_includes.err().unwrap()))
        }
        // todo: resolve other preprocessor directives
        Ok(resolved_includes.ok().unwrap())
    }
}

