#![feature(proc_macro_span)]
#![feature(proc_macro_diagnostic)]

use proc_macro::{Diagnostic, Level, Span, TokenStream, TokenTree};
use std::path::PathBuf;

use wgsl_preprocessor::WGSLPreprocessor;

fn read_file(file_path: PathBuf) -> String {
    match std::fs::read_to_string(file_path.clone()) {
        Ok(source) => source,
        Err(e) => {
            Diagnostic::spanned(
                Span::call_site(),
                Level::Error,
                format!("couldn't read {}: {}", file_path.display(), e),
            );
            panic!("Failed to preprocess WGSL source.");
        }
    }
}

#[proc_macro]
pub fn include_preprocessed(input: TokenStream) -> TokenStream {
    let mut tokens = input.into_iter();
    let file_path = match tokens.next() {
        Some(TokenTree::Literal(literal)) => {
            let str_literal = literal.to_string();
            literal.to_string().as_str()[1..str_literal.len() - 1].to_string()
        }
        _ => {
            panic!("Expected literal!")
        }
    };
    let call_site = Span::call_site();
    let mut own_path = call_site.source_file().path();
    assert!(own_path.pop());
    let new_path = own_path.join(&file_path);

    // todo: parse include_dirs & includes into WgslPreprocessor
    // todo: maybe define a config object that a WgslPreprocessor can be created from and that can be parsed from a JSON?
    for _token in tokens {
        //println!("{:?}", token);
    }

    let includes = [
        ("aabb", "aabb.wgsl"),
        ("bresenham", "bresenham.wgsl"),
        ("camera", "camera.wgsl"),
        ("constant", "constant.wgsl"),
        ("gpu_list", "gpu_list.wgsl"),
        ("page_table", "page_table.wgsl"),
        ("ray", "ray.wgsl"),
        ("sphere", "sphere.wgsl"),
        ("transform", "transform.wgsl"),
        ("type_alias", "type_alias.wgsl"),
        ("util", "util.wgsl"),
    ];

    let mut preprocessor = WGSLPreprocessor::default();
    for (id, path) in includes {
        let include_dir = own_path.join("renderer/wgsl");
        preprocessor.include(id.to_string(), read_file(include_dir.join(path)));
    }

    let src = read_file(new_path);
    let preprocessed = preprocessor.preprocess(src).unwrap();
    // todo: validate wgsl using naga if feature is activated

    println!("env: {}!!!", env!("WGSL_RELATIVE_INCLUDE_DIRS"));

    format!("{{ include_str!(\"{file_path}\"); {preprocessed:?} }}")
        .parse()
        .expect("Cannot format return preprocessed WGSL source")
}

// todo: maybe add macro that generates spv if feature is activated
