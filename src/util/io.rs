// adapted from https://github.com/asny/three-d/blob/master/src/io/loader.rs

use std::{
    collections::HashMap,
    path::PathBuf,
};
use reqwest::{
    Client,
    Error,
    Url,
};

pub async fn load_files_as_strings(mut paths: Vec<PathBuf>) -> Result<HashMap<PathBuf, String>, Error> {
    let mut result = HashMap::new();
    if paths.len() > 0 {
        let mut handles = Vec::new();
        let client = Client::new();
        for path in paths.drain(..) {
            let url = Url::parse(path.to_str().unwrap()).unwrap();
            handles.push((path, client.get(url).send().await));
        }
        for (path, handle) in handles.drain(..) {
            let string = handle
                .map_err(|e| e)?
                .text()
                .await
                .map_err(|e| e)?;
            result.insert(path, string);
        }
    }
    Ok(result)
}

pub async fn load_files_as_bytes(mut paths: Vec<PathBuf>) -> Result<HashMap<PathBuf, Vec<u8>>, Error> {
    let mut result = HashMap::new();
    if paths.len() > 0 {
        let mut handles = Vec::new();
        let client = Client::new();
        for path in paths.drain(..) {
            let url = Url::parse(path.to_str().unwrap()).unwrap();
            handles.push((path, client.get(url).send().await));
        }
        for (path, handle) in handles.drain(..) {
            let bytes = handle
                .map_err(|e| e)?
                .bytes()
                .await
                .map_err(|e| e)?
                .to_vec();
            result.insert(path, bytes);
        }
    }
    Ok(result)
}

pub async fn load_file_as_string(path: PathBuf) -> String {
    load_files_as_strings(vec!(path.clone()))
        .await
        .unwrap()
        .get(&path)
        .unwrap()
        .to_owned()
}

pub async fn load_file_as_bytes(path: PathBuf) -> Vec<u8> {
    load_files_as_bytes(vec!(path.clone()))
        .await
        .unwrap()
        .get(&path)
        .unwrap()
        .to_owned()
}
