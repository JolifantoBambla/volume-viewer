use crate::util::versioned;

pub mod v0_2;
pub mod v0_3;
pub mod v0_4;

versioned!(Multiscale {
   V0_4(v0_4::Multiscale : "0.4"),
   V0_3(v0_3::Multiscale : "0.3"),
   V0_2(v0_2::Multiscale : "0.2"),
});
