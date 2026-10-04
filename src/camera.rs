use crate::error::{LyraError, Result};
use crate::math::{look_at_matrix, matrix_to_quaternion, PluckerRay};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

/// Pinhole camera intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl CameraIntrinsics {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    /// Construct from horizontal and vertical field of view (in radians).
    pub fn from_fov(fov_x_rad: f32, fov_y_rad: f32, width: u32, height: u32) -> Self {
        let fx = (width as f32) / (2.0 * (fov_x_rad * 0.5).tan());
        let fy = (height as f32) / (2.0 * (fov_y_rad * 0.5).tan());
        let cx = (width as f32) * 0.5;
        let cy = (height as f32) * 0.5;
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    /// Construct default intrinsics for typical video/image resolution (e.g. 1280x720).
    pub fn default_for_resolution(width: u32, height: u32) -> Self {
        // Default horizontal FOV ~60 degrees
        let fov_x = 60.0_f32.to_radians();
        let aspect = (height as f32) / (width as f32);
        let fov_y = 2.0 * ((fov_x * 0.5).tan() * aspect).atan();
        Self::from_fov(fov_x, fov_y, width, height)
    }

    pub fn to_mat3(&self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.fx, 0.0, 0.0),
            Vec3::new(0.0, self.fy, 0.0),
            Vec3::new(self.cx, self.cy, 1.0),
        )
    }

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_cols(
            Vec4::new(self.fx, 0.0, 0.0, 0.0),
            Vec4::new(0.0, self.fy, 0.0, 0.0),
            Vec4::new(self.cx, self.cy, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn resized(&self, new_width: u32, new_height: u32) -> Self {
        let scale_x = (new_width as f32) / (self.width as f32);
        let scale_y = (new_height as f32) / (self.height as f32);
        Self {
            fx: self.fx * scale_x,
            fy: self.fy * scale_y,
            cx: self.cx * scale_x,
            cy: self.cy * scale_y,
            width: new_width,
            height: new_height,
        }
    }

    pub fn fov_x(&self) -> f32 {
        2.0 * ((self.width as f32 * 0.5) / self.fx).atan()
    }

    pub fn fov_y(&self) -> f32 {
        2.0 * ((self.height as f32 * 0.5) / self.fy).atan()
    }
}

/// Camera extrinsic pose parameters (Camera-to-World and World-to-Camera).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraExtrinsics {
    /// 4x4 Camera-to-World matrix.
    pub c2w: Mat4,
    /// 4x4 World-to-Camera matrix.
    pub w2c: Mat4,
}

impl CameraExtrinsics {
    pub fn from_c2w(c2w: Mat4) -> Self {
        let w2c = c2w.inverse();
        Self { c2w, w2c }
    }

    pub fn from_w2c(w2c: Mat4) -> Self {
        let c2w = w2c.inverse();
        Self { c2w, w2c }
    }

    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let w2c = look_at_matrix(eye, target, up, true);
        let c2w = w2c.inverse();
        Self { c2w, w2c }
    }

    pub fn position(&self) -> Vec3 {
        self.c2w.w_axis.truncate()
    }

    pub fn rotation_quat(&self) -> Quat {
        let rot_mat = Mat3::from_mat4(self.c2w);
        matrix_to_quaternion(&rot_mat)
    }

    pub fn forward(&self) -> Vec3 {
        // Forward vector in camera coordinates is Z
        self.c2w.z_axis.truncate().normalize()
    }

    pub fn up(&self) -> Vec3 {
        self.c2w.y_axis.truncate().normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.c2w.x_axis.truncate().normalize()
    }
}

/// Complete camera state combining extrinsics, intrinsics, and optional timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    pub extrinsics: CameraExtrinsics,
    pub intrinsics: CameraIntrinsics,
    pub frame_idx: usize,
    pub timestamp: f32,
}

impl CameraPose {
    pub fn new(
        extrinsics: CameraExtrinsics,
        intrinsics: CameraIntrinsics,
        frame_idx: usize,
        timestamp: f32,
    ) -> Self {
        Self {
            extrinsics,
            intrinsics,
            frame_idx,
            timestamp,
        }
    }

    /// Project a 3D point in world space to 2D image coordinates (u, v) and depth z.
    ///
    /// Returns Some((Vec2, depth)) if point is in front of the camera, None if behind near plane.
    pub fn project_point(&self, world_pt: Vec3) -> Option<(Vec2, f32)> {
        let pt4 = Vec4::new(world_pt.x, world_pt.y, world_pt.z, 1.0);
        let cam_pt = self.extrinsics.w2c * pt4;

        let z = cam_pt.z;
        if z <= 1e-4 {
            return None;
        }

        let u = self.intrinsics.fx * (cam_pt.x / z) + self.intrinsics.cx;
        let v = self.intrinsics.fy * (cam_pt.y / z) + self.intrinsics.cy;

        Some((Vec2::new(u, v), z))
    }

    /// Backproject a 2D pixel coordinate (u, v) with depth z into 3D world space.
    pub fn unproject_point(&self, u: f32, v: f32, depth: f32) -> Result<Vec3> {
        if depth <= 0.0 {
            return Err(LyraError::InvalidCamera(format!(
                "Depth must be positive for backprojection, got {}",
                depth
            )));
        }

        let x_cam = (u - self.intrinsics.cx) * depth / self.intrinsics.fx;
        let y_cam = (v - self.intrinsics.cy) * depth / self.intrinsics.fy;
        let z_cam = depth;

        let cam_pt = Vec4::new(x_cam, y_cam, z_cam, 1.0);
        let world_pt = self.extrinsics.c2w * cam_pt;
        Ok(world_pt.truncate())
    }

    /// Generates the ray in world space corresponding to pixel (u, v).
    pub fn ray_at_pixel(&self, u: f32, v: f32) -> (Vec3, Vec3) {
        let origin = self.extrinsics.position();
        let x_cam = (u - self.intrinsics.cx) / self.intrinsics.fx;
        let y_cam = (v - self.intrinsics.cy) / self.intrinsics.fy;
        let dir_cam = Vec3::new(x_cam, y_cam, 1.0).normalize();

        let rot = Mat3::from_mat4(self.extrinsics.c2w);
        let dir_world = (rot * dir_cam).normalize();
        (origin, dir_world)
    }

    /// Computes the Plücker coordinates for the ray at pixel (u, v).
    pub fn plucker_at_pixel(&self, u: f32, v: f32) -> PluckerRay {
        let (origin, dir) = self.ray_at_pixel(u, v);
        PluckerRay::new(origin, dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_camera_projection_and_unprojection() {
        let eye = Vec3::new(0.0, 1.0, 4.0);
        let target = Vec3::ZERO;
        let extrinsics = CameraExtrinsics::look_at(eye, target, Vec3::Y);
        let intrinsics = CameraIntrinsics::default_for_resolution(1280, 720);
        let camera = CameraPose::new(extrinsics, intrinsics, 0, 0.0);

        let test_pt = Vec3::new(0.5, 0.2, 0.1);
        let (screen_coords, depth) = camera
            .project_point(test_pt)
            .expect("Should project in front of camera");

        let reconstructed = camera
            .unproject_point(screen_coords.x, screen_coords.y, depth)
            .expect("Unproject should succeed");

        assert_relative_eq!(test_pt.x, reconstructed.x, epsilon = 1e-4);
        assert_relative_eq!(test_pt.y, reconstructed.y, epsilon = 1e-4);
        assert_relative_eq!(test_pt.z, reconstructed.z, epsilon = 1e-4);
    }

    #[test]
    fn test_intrinsics_scaling() {
        let orig = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 360.0, 1280, 720);
        let resized = orig.resized(640, 360);

        assert_relative_eq!(resized.fx, 500.0);
        assert_relative_eq!(resized.fy, 500.0);
        assert_relative_eq!(resized.cx, 320.0);
        assert_relative_eq!(resized.cy, 180.0);
    }
}
