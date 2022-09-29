use crate::error::*;

use regex::Regex;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct WGSLPreprocessor {
    include_files: HashMap<String, String>,
    include_regex: Regex,
}

impl Default for WGSLPreprocessor {
    fn default() -> Self {
        Self {
            include_files: HashMap::new(),
            include_regex: Regex::new(r"^@include\(\s*(?P<identifier>\S+)\s*\)")
                .expect("Could not construct regex"),
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
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.include_files.insert(identifier.into(), source.into());
        self
    }

    pub fn resolve_includes<S>(&self, source: S) -> Result<String, IncludeResolveError>
    where
        S: Into<String>,
    {
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
                        return Err(IncludeResolveError::RecursiveInclude(IncludeError::new(
                            line_number,
                            identifier.to_string(),
                            include_stack.pop().unwrap_or("source").to_string(),
                        )));
                    }

                    if let Some(source) = self.include_files.get(identifier) {
                        include_stack.push(&*identifier);
                        source_stack.push(lines);
                        source_stack.push(source.lines().enumerate());
                        resolved = false;
                        break;
                    } else {
                        return Err(IncludeResolveError::NotFound(IncludeError::new(
                            line_number,
                            identifier.to_string(),
                            include_stack.pop().unwrap_or("source").to_string(),
                        )));
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
    where
        S: Into<String>,
    {
        let resolved_includes = self.resolve_includes(source);
        if resolved_includes.is_err() {
            return Err(PreprocessorError::IncludeResolveError(
                resolved_includes.err().unwrap(),
            ));
        }

        // todo: resolve other preprocessor directives
        //  - @define()
        //  - @ifdef()
        //  - @ifndef()
        //  - @if()
        //  - @elif()
        //  - @else()
        //  - @endif()

        // todo: fix attributes for older target browser (version)
        //  - @<stagename> -> @stage(<stagename>)
        //  - const -> let

        Ok(resolved_includes.ok().unwrap())
    }
}

// todo: macro for compile time preprocessing

// todo: write test cases
#[cfg(test)]
mod tests {
    fn foo() -> &str {
        r"fn foo(x: f32) -> f32 { return x; }"
    }

    fn bar() -> &str {
        r"fn bar(x: f32) -> f32 { return x; }"
    }

    fn include_foo() -> &str {
        r"@include(foo.wgsl)"
    }

    fn include_bar() -> &str {
        r"@include(bar.wgsl)"
    }

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
