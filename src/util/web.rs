use std::path::PathBuf;
use wasm_bindgen::JsCast;
use web_sys::{Document, Element, HtmlCanvasElement, HtmlElement, Performance, Window};

#[inline]
pub fn window() -> Window {
    web_sys::window()
        .expect("window does not exist")
}

#[inline]
pub fn document() -> Document {
    window()
        .document()
        .expect("window has no document")
}

#[inline]
pub fn body() -> HtmlElement {
    document()
        .body()
        .expect("document has no body")
}

#[inline]
pub fn url() -> String {
    document()
        .url()
        .expect("document has no URL")
}

#[inline]
pub fn performance() -> Performance {
    window()
        .performance()
        .expect("window has no performance")
}

/// Returns the milliseconds since the HTML document's time origin (i.e the beginning of its lifetime).
/// Internally it calls `performance.now()`.
/// See the [JavaScript documentation](https://developer.mozilla.org/en-US/docs/Web/API/Performance/now) for more information.
pub fn now() -> f64 {
    performance()
        .now()
}

#[inline]
pub fn get_element_by_id(id: &str) -> Element {
    document()
        .get_element_by_id(id)
        .expect(format!("document has no element with id {}", id).to_string().as_str())
}

#[inline]
pub fn get_canvas_by_id(id: &str) -> HtmlCanvasElement {
    get_element_by_id(id)
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| ())
        .expect(format!("element with id {} was no canvas", id).to_string().as_str())
}

/// Attaches a given HtmlCanvasElement to the document.
/// If a parent_id is given, the canvas is appended as child of the parent element.
/// Otherwise, the canvas as attached to the body of the document.
#[inline]
pub fn attach_canvas(canvas: HtmlCanvasElement, parent_id: Option<String>) {
    let parent = if parent_id.is_none() {
        body()
            .dyn_into::<HtmlElement>()
            .map_err(|_| ())
            .unwrap()
    } else {
        get_element_by_id(parent_id.unwrap().as_str())
            .dyn_into::<HtmlElement>()
            .map_err(|_| ())
            .unwrap()
    };
    parent.append_child(&web_sys::Element::from(canvas))
        .ok()
        .expect("could not append element to document");
}

#[inline]
pub fn base_path() -> PathBuf {
    let base_url = url();
    if !base_url.ends_with('/') {
        PathBuf::from(base_url).parent().unwrap().to_path_buf()
    } else {
        PathBuf::from(base_url)
    }
}
