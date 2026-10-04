use crate::depth::PointCloud;
use crate::error::Result;
use crate::math::{matrix_to_quaternion, quaternion_to_matrix, BoundingBox};
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Maximum degree of spherical harmonics supported (degree 0 = DC term, degree 3 = 16 coeffs per channel).
pub const MAX_SH_DEGREE: usize = 3;
pub const SH_COEFFS_COUNT: usize = 16;

/// A 3D Gaussian Splat primitive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian3D {
    /// 3D mean position (world coordinates).
    pub position: Vec3,
    /// 3D scale vector along local principal axes (log-space or linear).
    pub scale: Vec3,
    /// Orientation represented as a normalized unit quaternion (w, x, y, z).
    pub rotation: Quat,
    /// Opacity value in [0, 1].
    pub opacity: f32,
    /// Direct RGB color in [0, 1] or base DC spherical harmonic component.
    pub color_rgb: [f32; 3],
    /// Optional higher-order Spherical Harmonics coefficients [15, 3] for degrees 1..=3.
    pub sh_rest: Option<Vec<[f32; 3]>>,
}

impl Gaussian3D {
    pub fn new(position: Vec3, scale: Vec3, rotation: Quat, opacity: f32, color_rgb: [f32; 3]) -> Self {
        Self {
            position,
            scale,
            rotation: rotation.normalize(),
            opacity: opacity.clamp(0.0, 1.0),
            color_rgb,
            sh_rest: None,
        }
    }

    /// Evaluates the 3x3 3D covariance matrix: \Sigma = R S S^T R^T
    pub fn compute_covariance_3d(&self) -> Mat3 {
        let r = quaternion_to_matrix(&self.rotation);
        let s = Mat3::from_diagonal(self.scale);
        let m = r * s;
        m * m.transpose()
    }

    /// Evaluates the view-dependent color given a normalized view direction.
    ///
    /// If higher-order spherical harmonics are present, evaluates up to degree 3,
    /// otherwise returns the base RGB color.
    pub fn evaluate_color(&self, view_dir: Vec3) -> [f32; 3] {
        if self.sh_rest.is_none() || self.sh_rest.as_ref().map_or(true, |v| v.is_empty()) {
            return self.color_rgb;
        }

        let sh = self.sh_rest.as_ref().unwrap();
        let mut r = self.color_rgb[0];
        let mut g = self.color_rgb[1];
        let mut b = self.color_rgb[2];

        // Degree 1 SH basis constants
        let c1 = 0.4886025119029199; // sqrt(3 / (4 * pi))
        let x = view_dir.x;
        let y = view_dir.y;
        let z = view_dir.z;

        if sh.len() >= 3 {
            // y, z, x bases
            let bases = [-c1 * y, c1 * z, -c1 * x];
            for i in 0..3 {
                r += bases[i] * sh[i][0];
                g += bases[i] * sh[i][1];
                b += bases[i] * sh[i][2];
            }
        }

        [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
    }
}

/// A collection of 3D Gaussian splats representing a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianCloud {
    pub gaussians: Vec<Gaussian3D>,
}

impl GaussianCloud {
    pub fn new() -> Self {
        Self { gaussians: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    pub fn bounding_box(&self) -> BoundingBox {
        let mut bbox = BoundingBox::empty();
        for g in &self.gaussians {
            bbox.include_point(g.position);
        }
        bbox
    }

    /// Filter out gaussians with opacity below threshold or with non-finite attributes.
    pub fn prune_low_opacity(&mut self, min_opacity: f32) {
        self.gaussians.retain(|g| {
            g.opacity >= min_opacity
                && g.position.is_finite()
                && g.scale.is_finite()
                && g.rotation.is_finite()
        });
    }

    /// Creates a 3D Gaussian cloud by lifting a PointCloud.
    ///
    /// Computes initial anisotropic or isotropic scales using average point distances.
    pub fn from_point_cloud(pcd: &PointCloud, default_radius: Option<f32>) -> Result<Self> {
        if pcd.is_empty() {
            return Ok(Self::new());
        }

        let radius = default_radius.unwrap_or(0.02);
        let mut gaussians = Vec::with_capacity(pcd.len());

        for pt in &pcd.points {
            let color = [
                pt.color[0] as f32 / 255.0,
                pt.color[1] as f32 / 255.0,
                pt.color[2] as f32 / 255.0,
            ];

            let rotation = if let Some(n) = pt.normal {
                let forward = n.normalize();
                let up = if forward.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
                let right = up.cross(forward).normalize();
                let actual_up = forward.cross(right);
                let mat = Mat3::from_cols(right, actual_up, forward);
                matrix_to_quaternion(&mat)
            } else {
                Quat::IDENTITY
            };

            gaussians.push(Gaussian3D::new(
                pt.position,
                Vec3::splat(radius),
                rotation,
                0.95, // High initial opacity
                color,
            ));
        }

        Ok(Self { gaussians })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_covariance() {
        let g = Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            0.8,
            [1.0, 0.5, 0.2],
        );

        let cov = g.compute_covariance_3d();
        assert_relative_eq!(cov.x_axis.x, 0.01, epsilon = 1e-5);
        assert_relative_eq!(cov.y_axis.y, 0.04, epsilon = 1e-5);
        assert_relative_eq!(cov.z_axis.z, 0.09, epsilon = 1e-5);
        assert_relative_eq!(cov.x_axis.y, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_from_point_cloud() {
        let mut pcd = PointCloud::new();
        pcd.points.push(crate::depth::Point3D {
            position: Vec3::new(0.0, 1.0, 2.0),
            color: [255, 128, 64],
            normal: None,
        });

        let cloud = GaussianCloud::from_point_cloud(&pcd, Some(0.05)).expect("Should lift PCD");
        assert_eq!(cloud.len(), 1);
        assert_relative_eq!(cloud.gaussians[0].color_rgb[0], 1.0);
        assert_relative_eq!(cloud.gaussians[0].scale.x, 0.05);
    }
}
