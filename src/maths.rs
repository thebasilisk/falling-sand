use std::ffi::c_float;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Float2(pub c_float, pub c_float);
