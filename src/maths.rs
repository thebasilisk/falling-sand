use std::{ffi::c_float, ops::Index};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Float4(pub c_float,pub c_float, pub c_float, pub c_float);
impl Float4 {
    pub fn _new(v1 : Float2, v2 : Float2) -> Self {
        Self(v1.0, v1.1, v2.0, v2.1)
    }
    pub fn from_float3(v : Float3, s : f32) -> Self {
        Self(v.0, v.1, v.2, s)
    }
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Float3(pub c_float, pub c_float, pub c_float);
impl Float3 {
    pub fn new(v : Float2, f : f32) -> Self {
        Self(v.0, v.1, f)
    }
    pub fn magnitude(&self) -> f32 {
        (self.0.powf(2.0) + self.1.powf(2.0) + self.2.powf(2.0)).sqrt()
    }
    pub fn normalized(&self) -> Float3 {
        Float3(self.0 / self.magnitude(), self.1 / self.magnitude(), self.2 / self.magnitude())
    }
    pub fn single(f : f32) -> Self {
        Float3(f, f, f)
    }
    pub fn fminf(&self, rhs : Float3) -> Float3 {
        Float3(self.0.min(rhs.0), self.1.min(rhs.1), self.2.min(rhs.2))
    }
    pub fn fmaxf(&self, rhs : Float3) -> Float3 {
        Float3(self.0.max(rhs.0), self.1.max(rhs.1), self.2.max(rhs.2))
    }
}
impl Index<usize> for Float3 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => &self.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Float2(pub c_float, pub c_float);
impl Float2 {
    pub fn magnitude(&self) -> f32 {
        //add cached result
        (self.0.powf(2.0) + self.1.powf(2.0)).sqrt()
    }
    pub fn normalized(&self) -> Float2 {
        Float2(self.0 / self.magnitude(), self.1 / self.magnitude())
    }
    pub fn default() -> Float2 {
        Float2(0.0, 0.0)
    }
}

pub struct Float2x2 {
    row1 : Float2,
    row2 : Float2
}

#[inline]
pub fn float2_add(v1 : Float2, v2 : Float2) -> Float2 {
    Float2(v1.0 + v2.0, v1.1 + v2.1)
}

#[inline]
pub fn float2_subtract(v1 : Float2, v2 : Float2) -> Float2 {
    Float2(v1.0 - v2.0, v1.1 - v2.1)
}
#[inline]
pub fn scale2 (v : Float2, f : f32) -> Float2 {
    Float2(v.0 * f, v.1 * f)
}

#[inline]
pub fn float3_add(v1 : Float3, v2 : Float3) -> Float3 {
    Float3(v1.0 + v2.0, v1.1 + v2.1, v1.2 + v2.2)
}

#[inline]
pub fn float3_subtract(v1 : Float3, v2 : Float3) -> Float3 {
    Float3(v1.0 - v2.0, v1.1 - v2.1, v1.2 - v2.2)
}
#[inline]
pub fn scale3 (v : Float3, f : f32) -> Float3 {
    Float3(v.0 * f, v.1 * f, v.2 * f)
}

#[inline]
pub fn dot(v1 : Float2, v2 : Float2) -> f32 {
    v1.0 * v2.0 + v1.1 * v2.1
}

#[inline]
pub fn dot3(v1 : Float3, v2 : Float3) -> f32 {
    v1.0 * v2.0 + v1.1 * v2.1 + v1.2 * v2.2
}

#[inline]
pub fn matrix_mul(v : Float2, m : Float2x2) -> Float2 {
    Float2(dot(v, m.row1), dot(v, m.row2))
}

#[inline]
pub fn rotation_matrix(theta : f32) -> Float2x2 {
    let cos_theta = f32::cos(theta);
    let sin_theta = f32::sin(theta);
    Float2x2 {
        row1 : Float2(cos_theta, -sin_theta),
        row2 : Float2(sin_theta, cos_theta)
    }
}

#[inline]
pub fn apply_rotation_float2(target : Float2, theta : f32) -> Float2 {
    matrix_mul(target, rotation_matrix(theta))
}

#[inline]
pub fn cross_product(vec1 : &Float3, vec2 : &Float3) -> Float3 {
    Float3(
        vec1.1 * vec2.2 - vec1.2 * vec2.1,
        vec1.2 * vec2.0 - vec1.0 * vec2.2,
        vec1.0 * vec2.1 - vec1.1 * vec2.0
    )
}


pub fn calculate_quaternion(camera_rotation_dir : &Float3) -> Float4 {
    let default_rotation = Float3(0.0, 0.0, 1.0);
    let camera_rotation = camera_rotation_dir.normalized();

    let rotation_axis = cross_product(&default_rotation, &camera_rotation);
    let rotation_axis_normalized = rotation_axis.normalized();

    //assuming normalized camera rotation vector
    let half_theta = rotation_axis.magnitude().asin() / 2.0;
    //println!("{}", half_theta / PI);

    Float4(
        rotation_axis_normalized.0 * half_theta.sin(),
        rotation_axis_normalized.1 * half_theta.sin(),
        rotation_axis_normalized.2 * half_theta.sin(),
        half_theta.cos()
    )
}


pub fn update_quat_angle(q : &Float4, theta : f32) -> Float4 {
    let new_ratio = theta.sin() / q.3.acos().sin();
    Float4(q.0 * new_ratio, q.1 * new_ratio, q.2 * new_ratio, theta.cos())
}


fn quat_inv(quat : Float4) -> Float4 {
    Float4(-quat.0, -quat.1, -quat.2, quat.3)
}

fn quat_dot (q1 : Float4, q2 : Float4) -> Float4 {
    let s : f32 = q1.3 * q2.3 - dot3(Float3(q1.0, q1.1, q1.2), Float3(q2.0, q2.1, q2.2));
    let v : Float3 = float3_add(cross_product(&Float3(q1.0, q1.1, q1.2), &Float3(q2.0, q2.1, q2.2)), float3_add(scale3(Float3(q2.0, q2.1, q2.2), q1.3), scale3(Float3(q1.0, q1.1, q1.2), q2.3)));
    Float4::from_float3(v, s)
}

pub fn quat_mult (vec : Float3, quat : Float4) -> Float3 {
    let r = quat_dot(quat_dot(quat_inv(quat), Float4::from_float3(vec, 0.0)), quat);
    Float3(r.0, r.1, r.2)
}
