use std::collections::{HashMap, HashSet};
use regex::Regex;

pub fn resolve_includes() {
    const STDLIB: HashMap<&str, &str> = HashMap::from([
        ("utl::type_alias", include_str!("type_alias.wgsl"))
    ]);
}



// todo: define errors
#[derive(Debug, Clone)]
pub struct PreprocessorError {}

const INCLUDE_REGEX: &str = "^@include\((?P<file>\S+)\)";

#[derive(Debug, Default)]
pub struct WGSLPreprocessor {
    include_files: HashMap<String, String>,
    include_regex: Regex,
}

impl WGSLPreprocessor {
    pub fn new() -> Self {
        WGSLPreprocessor {
            include_regex: Regex::new(INCLUDE_REGEX).expect("Could not construct regex"),
            ..Default::default()
        }
    }

    pub fn include<K, V>(&mut self, identifier: K, source: V) -> &mut Self
    where K: Into<String>, V: Into<String>
    {
        self.include_files.insert(identifier.into(), source.into());
        self
    }

    pub fn resolve_include<S>(&self, source: S) -> Result<String, PreprocessorError>
    where S: Into<String>
    {
        Ok("".to_string())
    }

    fn expand_recursive(
        &self,
        expanded_src: &mut Vec<String>,
        src: &str,
        in_file: Option<&str>,
        include_stack: &mut Vec<&str>,
        include_set: &mut HashSet<&str>,
    ) -> Result<(), PreprocessorError> {
        let mut need_line_directive = false;
        // Iterate through each line in the src input
        // - If the line matches our INCLUDE_RE regex, recurse
        // - Otherwise, add the line to our outputs and continue to the next line
        for (line_num, line) in src.lines().enumerate() {
            if let Some(caps) = self.include_regex.captures(line) {
                // The following expect should be impossible, but write a nice message anyways
                let cap_match = caps
                    .name("file")
                    .expect("Could not find capture group with name \"file\"");
                let included_file = cap_match.as_str();

                // if this file has already been included, continue to the next line
                // this acts as a header guard
                if include_set.contains(&included_file) {
                    continue;
                }

                // return if the included file already exists in the include_stack
                // this signals that we're in an infinite loop
                if include_stack.contains(&included_file) {
                    let in_file = in_file.map(|s| s.to_string());
                    let problem_include = included_file.to_string();
                    let include_stack = include_stack.into_iter().map(|s| s.to_string()).collect();
                    return Err(Error::RecursiveInclude {
                        in_file: in_file,
                        line_num: line_num,
                        problem_include: problem_include,
                        include_stack: include_stack,
                    });
                }

                // if the included file exists in our context, recurse
                if let Some(src) = self.included_files.get(included_file) {
                    include_stack.push(&included_file);
                    self.expand_recursive(
                        expanded_src,
                        &src,
                        Some(included_file),
                        include_stack,
                        include_set,
                    )?;
                    include_stack.pop();
                    need_line_directive = true;
                } else {
                    let in_file = in_file.map(|s| s.to_string());
                    let problem_include = included_file.to_string();
                    return Err(Error::FileNotFound {
                        in_file: in_file,
                        line_num: line_num,
                        problem_include: problem_include,
                    });
                }
            } else {
                // Got a regular line
                if need_line_directive {
                    // add a #line directive to reset the line number so that GL compilation error
                    // messages contain line numbers that map to the users file
                    expanded_src.push(format!("#line {} 0", line_num + 1));
                }
                need_line_directive = false;
                expanded_src.push(String::from(line));
            }
        }

        // Add the in_file to the include set to prevent
        // future inclusions
        if let Some(in_file) = in_file {
            include_set.insert(in_file);
        }
        Ok(())
    }
}

