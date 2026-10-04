use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};

/// 3D Axis-Aligned Bounding Box (AABB).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    pub fn include_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        (self.max - self.min).max(Vec3::ZERO)
    }

    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        let closest = center.clamp(self.min, self.max);
        closest.distance_squared(center) <= radius * radius
    }
}

/// Plücker line coordinate representation for camera ray conditioning.
///
/// In Project Lyra, camera rays are encoded as normalized Plücker coordinates:
/// L = (d, o x d), where d is the unit ray direction, and (o x d) is the ray moment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PluckerRay {
    pub direction: Vec3,
    pub moment: Vec3,
}

impl PluckerRay {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        let d = direction.normalize();
        let moment = origin.cross(d);
        Self {
            direction: d,
            moment,
        }
    }

    pub fn as_array(&self) -> [f32; 6] {
        [
            self.direction.x,
            self.direction.y,
            self.direction.z,
            self.moment.x,
            self.moment.y,
            self.moment.z,
        ]
    }
}

/// Creates a 4x4 Look-At view matrix (World-to-Camera or Camera-to-World).
///
/// In standard graphics/vision conventions (Y-up, camera looking toward target):
/// - forward: normalized vector pointing from camera to target
/// - right: cross(up, forward)
/// - up: cross(forward, right)
pub fn look_at_matrix(camera_pos: Vec3, target: Vec3, up_hint: Vec3, world_to_camera: bool) -> Mat4 {
    let forward = (target - camera_pos).normalize_or_zero();
    let right = up_hint.cross(forward).normalize_or_zero();
    let up = forward.cross(right).normalize_or_zero();

    let mut mat = Mat4::IDENTITY;
    mat.x_axis = Vec4::new(right.x, up.x, forward.x, 0.0);
    mat.y_axis = Vec4::new(right.y, up.y, forward.y, 0.0);
    mat.z_axis = Vec4::new(right.z, up.z, forward.z, 0.0);

    if world_to_camera {
        // Translation: t = -R * C
        let r = Mat3::from_cols(right, up, forward);
        let t = -r * camera_pos;
        mat.w_axis = Vec4::new(t.x, t.y, t.z, 1.0);
    } else {
        mat.w_axis = Vec4::new(camera_pos.x, camera_pos.y, camera_pos.z, 1.0);
    }

    mat
}

/// Converts a 3x3 rotation matrix to a normalized quaternion.
pub fn matrix_to_quaternion(mat: &Mat3) -> Quat {
    Quat::from_mat3(mat).normalize()
}

/// Converts a quaternion to a 3x3 rotation matrix.
pub fn quaternion_to_matrix(quat: &Quat) -> Mat3 {
    Mat3::from_quat(quat.normalize())
}

/// Spherical Linear Interpolation (SLERP) between two quaternions.
pub fn slerp(q1: Quat, q2: Quat, t: f32) -> Quat {
    q1.slerp(q2, t).normalize()
}

/// Evaluates a Catmull-Rom spline at parameter t in [0, 1] across 4 control points.
pub fn catmull_rom_spline(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;

    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_look_at_and_quaternion() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let target = Vec3::ZERO;
        let up = Vec3::Y;

        let w2c = look_at_matrix(eye, target, up, true);
        assert_relative_eq!(w2c.w_axis.w, 1.0);

        let c2w = look_at_matrix(eye, target, up, false);
        assert_relative_eq!(c2w.w_axis.x, 0.0);
        assert_relative_eq!(c2w.w_axis.y, 0.0);
        assert_relative_eq!(c2w.w_axis.z, 5.0);

        let rot = Mat3::from_mat4(c2w);
        let quat = matrix_to_quaternion(&rot);
        let back_rot = quaternion_to_matrix(&quat);

        for col in 0..3 {
            let v1 = rot.col(col);
            let v2 = back_rot.col(col);
            assert_relative_eq!(v1.x, v2.x, epsilon = 1e-4);
            assert_relative_eq!(v1.y, v2.y, epsilon = 1e-4);
            assert_relative_eq!(v1.z, v2.z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_plucker_ray() {
        let origin = Vec3::new(1.0, 2.0, 3.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);
        let plucker = PluckerRay::new(origin, direction);

        assert_relative_eq!(plucker.direction.length(), 1.0);
        assert_relative_eq!(plucker.moment.x, -2.0);
        assert_relative_eq!(plucker.moment.y, 1.0);
        assert_relative_eq!(plucker.moment.z, 0.0);
    }

    #[test]
    fn test_catmull_rom_spline() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(1.0, 0.0, 0.0);
        let p2 = Vec3::new(2.0, 1.0, 0.0);
        let p3 = Vec3::new(3.0, 1.0, 0.0);

        let start = catmull_rom_spline(p0, p1, p2, p3, 0.0);
        let end = catmull_rom_spline(p0, p1, p2, p3, 1.0);

        assert_relative_eq!(start.x, 1.0, epsilon = 1e-5);
        assert_relative_eq!(start.y, 0.0, epsilon = 1e-5);
        assert_relative_eq!(end.x, 2.0, epsilon = 1e-5);
        assert_relative_eq!(end.y, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_bounding_box_intersection() {
        let mut bbox = BoundingBox::empty();
        bbox.include_point(Vec3::new(-1.0, -1.0, -1.0));
        bbox.include_point(Vec3::new(1.0, 1.0, 1.0));

        assert!(bbox.contains(Vec3::new(0.5, 0.0, -0.5)));
        assert!(!bbox.contains(Vec3::new(2.0, 0.0, 0.0)));
        assert!(bbox.intersects_sphere(Vec3::new(1.5, 0.0, 0.0), 0.6));
        assert!(!bbox.intersects_sphere(Vec3::new(3.0, 0.0, 0.0), 0.5));
    }
}
