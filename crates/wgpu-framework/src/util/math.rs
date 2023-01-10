pub mod f64 {
    pub const SQRT_5: f64 = 2.236_067_977_499_79_f64;

    /// The [golden ratio](https://en.wikipedia.org/wiki/Golden_ratio).
    pub const PHI: f64 = (1.0 + SQRT_5) / 2.0;

    #[inline]
    pub fn is_close_to_zero(val: f64) -> bool {
        val.abs() < f64::EPSILON
    }
}

pub mod f32 {
    pub const SQRT_5: f32 = crate::util::math::f64::SQRT_5 as f32;

    /// The [golden ratio](https://en.wikipedia.org/wiki/Golden_ratio).
    pub const PHI: f32 = (1.0 + SQRT_5) / 2.0;

    #[inline]
    pub fn is_close_to_zero(val: f32) -> bool {
        val.abs() < f32::EPSILON
    }
}
