use crate::util::versioned;

pub mod v0_1;
pub mod v0_2;
pub mod v0_3 {
    pub use crate::image_label::v0_2::*;
}
pub mod v0_4 {
    pub use crate::image_label::v0_2::*;
}

// Note: "image-label.version" is never mentioned in the spec, so I treat it as optional, i.e. it
// needs util::versioned

versioned!(ImageLabel {
   V0_4(v0_4::ImageLabel : "0.4"),
   V0_3(v0_3::ImageLabel : "0.3"),
   V0_2(v0_2::ImageLabel : "0.2"),
   V0_1(v0_1::ImageLabel : "0.1"),
});
