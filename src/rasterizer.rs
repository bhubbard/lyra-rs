use crate::camera::CameraPose;
use crate::depth::DepthMap;
use crate::error::{LyraError, Result};
use crate::gaussian::GaussianCloud;
use glam::{Mat3, Vec2, Vec3, Vec4};
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;

/// Configuration for Gaussian Splatting rasterizer.
#[derive(Debug, Clone)]
pub struct RasterizerConfig {
    pub bg_color: [f32; 3],
    pub near_plane: f32,
    pub far_plane: f32,
    pub tile_size: u32,
    pub max_splat_radius: f32,
}

impl Default for RasterizerConfig {
    fn default() -> Self {
        Self {
            bg_color: [0.0, 0.0, 0.0],
            near_plane: 0.1,
            far_plane: 100.0,
            tile_size: 16,
            max_splat_radius: 512.0,
        }
    }
}

/// A projected 2D Gaussian primitive ready for screen rasterization.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedGaussian {
    pub screen_pos: Vec2,
    pub depth: f32,
    pub radius: f32,
    /// Conic parameters: A * dx^2 + 2 * B * dx * dy + C * dy^2
    pub conic: Vec3, // (A, B, C)
    pub opacity: f32,
    pub color: [f32; 3],
}

/// Rendered frame results containing RGB color buffer, expected depth map, and alpha mask.
pub struct RenderOutput {
    pub image: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub depth: DepthMap,
    pub alpha: Vec<f32>,
}

/// Pure Rust high-performance multi-threaded 3D Gaussian Splatting rasterizer.
pub struct GaussianRasterizer {
    pub config: RasterizerConfig,
}

impl GaussianRasterizer {
    pub fn new(config: RasterizerConfig) -> Self {
        Self { config }
    }

    /// Projects 3D Gaussians into screen space for a given camera pose.
    pub fn project_gaussians(
        &self,
        cloud: &GaussianCloud,
        camera: &CameraPose,
    ) -> Vec<ProjectedGaussian> {
        let w = camera.intrinsics.width as f32;
        let h = camera.intrinsics.height as f32;
        let fx = camera.intrinsics.fx;
        let fy = camera.intrinsics.fy;
        let cx = camera.intrinsics.cx;
        let cy = camera.intrinsics.cy;

        let w2c = camera.extrinsics.w2c;
        let w2c_rot = Mat3::from_mat4(w2c);

        let mut projected: Vec<ProjectedGaussian> = cloud
            .gaussians
            .par_iter()
            .filter_map(|g| {
                // Transform position to camera space
                let p_world = Vec4::new(g.position.x, g.position.y, g.position.z, 1.0);
                let p_cam = w2c * p_world;

                let z = p_cam.z;
                if z < self.config.near_plane || z > self.config.far_plane {
                    return None;
                }

                // Project to screen space
                let u = fx * (p_cam.x / z) + cx;
                let v = fy * (p_cam.y / z) + cy;

                // Frustum culling check with margin
                let margin = 100.0;
                if u < -margin || u > w + margin || v < -margin || v > h + margin {
                    return None;
                }

                // Compute 3D Covariance
                let cov3d = g.compute_covariance_3d();

                // Compute Jacobian J of projective transformation at p_cam
                // J = [fx / z,       0, -fx * x / z^2]
                //     [     0,  fy / z, -fy * y / z^2]
                let inv_z = 1.0 / z;
                let inv_z2 = inv_z * inv_z;

                let j00 = fx * inv_z;
                let j02 = -fx * p_cam.x * inv_z2;
                let j11 = fy * inv_z;
                let j12 = -fy * p_cam.y * inv_z2;

                // Rotate covariance into camera space: Cov_cam = W_rot * Cov3d * W_rot^T
                let cov_cam = w2c_rot * cov3d * w2c_rot.transpose();

                // 2D Screen covariance: Cov2D = J * Cov_cam * J^T
                let c00 = cov_cam.x_axis.x;
                let c01 = cov_cam.x_axis.y;
                let c02 = cov_cam.x_axis.z;
                let c11 = cov_cam.y_axis.y;
                let c12 = cov_cam.y_axis.z;
                let c22 = cov_cam.z_axis.z;

                let mut s00 = j00 * (j00 * c00 + j02 * c02) + j02 * (j00 * c02 + j02 * c22);
                let s01 = j00 * (j11 * c01 + j12 * c02) + j02 * (j11 * c12 + j12 * c22);
                let mut s11 = j11 * (j11 * c11 + j12 * c12) + j12 * (j11 * c12 + j12 * c22);

                // Low-pass filter to prevent aliasing (standard 3DGS 0.3 pixel blur)
                s00 += 0.3;
                s11 += 0.3;

                // Determinant of 2D covariance
                let det = s00 * s11 - s01 * s01;
                if det <= 1e-6 {
                    return None;
                }

                let inv_det = 1.0 / det;
                let conic_a = s11 * inv_det;
                let conic_b = -s01 * inv_det;
                let conic_c = s00 * inv_det;

                // Splat bounding radius from eigenvalues (3 sigma ~ 99% mass)
                let mid = 0.5 * (s00 + s11);
                let term = (0.25 * (s00 - s11) * (s00 - s11) + s01 * s01).sqrt();
                let lambda1 = mid + term;
                let radius = (3.0 * lambda1.sqrt()).clamp(1.0, self.config.max_splat_radius);

                // View direction from camera position to Gaussian center
                let view_dir = (g.position - camera.extrinsics.position()).normalize_or_zero();
                let color = g.evaluate_color(view_dir);

                Some(ProjectedGaussian {
                    screen_pos: Vec2::new(u, v),
                    depth: z,
                    radius,
                    conic: Vec3::new(conic_a, conic_b, conic_c),
                    opacity: g.opacity,
                    color,
                })
            })
            .collect();

        // Sort front-to-back by camera depth
        projected.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));
        projected
    }

    /// Rasterizes the Gaussian cloud into color, depth, and alpha buffers.
    pub fn render(&self, cloud: &GaussianCloud, camera: &CameraPose) -> Result<RenderOutput> {
        let width = camera.intrinsics.width;
        let height = camera.intrinsics.height;

        let projected = self.project_gaussians(cloud, camera);

        let mut img_buffer = vec![0u8; (width * height * 3) as usize];
        let mut depth_buffer = vec![0.0f32; (width * height) as usize];
        let mut alpha_buffer = vec![0.0f32; (width * height) as usize];

        // Tile-based or scanline parallel rendering
        let tile_size = self.config.tile_size as usize;
        let num_tiles_x = (width as usize + tile_size - 1) / tile_size;
        let num_tiles_y = (height as usize + tile_size - 1) / tile_size;
        let total_tiles = num_tiles_x * num_tiles_y;

        // Process tiles in parallel using Rayon
        let tile_results: Vec<(usize, Vec<u8>, Vec<f32>, Vec<f32>)> = (0..total_tiles)
            .into_par_iter()
            .map(|tile_idx| {
                let tx = tile_idx % num_tiles_x;
                let ty = tile_idx / num_tiles_x;

                let min_x = tx * tile_size;
                let max_x = ((tx + 1) * tile_size).min(width as usize);
                let min_y = ty * tile_size;
                let max_y = ((ty + 1) * tile_size).min(height as usize);

                let tile_w = max_x - min_x;
                let tile_h = max_y - min_y;

                let mut tile_rgb = vec![0u8; tile_w * tile_h * 3];
                let mut tile_depth = vec![0.0f32; tile_w * tile_h];
                let mut tile_alpha = vec![0.0f32; tile_w * tile_h];

                // Filter gaussians intersecting this tile
                let tile_min = Vec2::new(min_x as f32, min_y as f32);
                let tile_max = Vec2::new(max_x as f32, max_y as f32);

                let overlapping: Vec<&ProjectedGaussian> = projected
                    .iter()
                    .filter(|g| {
                        let rad = g.radius;
                        g.screen_pos.x + rad >= tile_min.x
                            && g.screen_pos.x - rad <= tile_max.x
                            && g.screen_pos.y + rad >= tile_min.y
                            && g.screen_pos.y - rad <= tile_max.y
                    })
                    .collect();

                // Rasterize each pixel in this tile
                for y in min_y..max_y {
                    let local_y = y - min_y;
                    for x in min_x..max_x {
                        let local_x = x - min_x;
                        let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);

                        let mut r = self.config.bg_color[0];
                        let mut g_col = self.config.bg_color[1];
                        let mut b = self.config.bg_color[2];
                        let mut acc_depth = 0.0f32;
                        let mut acc_alpha = 0.0f32;
                        let mut transmittance = 1.0f32;

                        for g in &overlapping {
                            let d = p - g.screen_pos;
                            let dist_sq = d.length_squared();
                            if dist_sq > g.radius * g.radius {
                                continue;
                            }

                            // Evaluate Gaussian power: -0.5 * (A * dx^2 + 2 * B * dx * dy + C * dy^2)
                            let power = -0.5
                                * (g.conic.x * d.x * d.x
                                    + 2.0 * g.conic.y * d.x * d.y
                                    + g.conic.z * d.y * d.y);

                            if power > 0.0 || power < -9.0 {
                                continue;
                            }

                            let alpha = (g.opacity * power.exp()).clamp(0.0, 0.99);
                            if alpha < 1.0 / 255.0 {
                                continue;
                            }

                            let weight = alpha * transmittance;
                            r += g.color[0] * weight;
                            g_col += g.color[1] * weight;
                            b += g.color[2] * weight;
                            acc_depth += g.depth * weight;
                            acc_alpha += weight;

                            transmittance *= 1.0 - alpha;
                            if transmittance < 0.0001 {
                                break; // Early ray termination
                            }
                        }

                        let p_idx = (local_y * tile_w + local_x) * 3;
                        tile_rgb[p_idx] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                        tile_rgb[p_idx + 1] = (g_col.clamp(0.0, 1.0) * 255.0) as u8;
                        tile_rgb[p_idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;

                        let d_idx = local_y * tile_w + local_x;
                        tile_depth[d_idx] = if acc_alpha > 0.0 {
                            acc_depth / acc_alpha
                        } else {
                            0.0
                        };
                        tile_alpha[d_idx] = acc_alpha.clamp(0.0, 1.0);
                    }
                }

                (tile_idx, tile_rgb, tile_depth, tile_alpha)
            })
            .collect();

        // Assemble tiles into full frame buffers
        for (tile_idx, tile_rgb, tile_depth, tile_alpha) in tile_results {
            let tx = tile_idx % num_tiles_x;
            let ty = tile_idx / num_tiles_x;

            let min_x = tx * tile_size;
            let max_x = ((tx + 1) * tile_size).min(width as usize);
            let min_y = ty * tile_size;
            let max_y = ((ty + 1) * tile_size).min(height as usize);

            let tile_w = max_x - min_x;

            for y in min_y..max_y {
                let local_y = y - min_y;
                let full_row_start = (y * width as usize + min_x) * 3;
                let tile_row_start = local_y * tile_w * 3;
                let row_bytes = tile_w * 3;

                img_buffer[full_row_start..full_row_start + row_bytes]
                    .copy_from_slice(&tile_rgb[tile_row_start..tile_row_start + row_bytes]);

                let full_d_start = y * width as usize + min_x;
                let tile_d_start = local_y * tile_w;
                depth_buffer[full_d_start..full_d_start + tile_w]
                    .copy_from_slice(&tile_depth[tile_d_start..tile_d_start + tile_w]);

                alpha_buffer[full_d_start..full_d_start + tile_w]
                    .copy_from_slice(&tile_alpha[tile_d_start..tile_d_start + tile_w]);
            }
        }

        let image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, img_buffer)
            .ok_or_else(|| LyraError::GaussianError("Failed to create image buffer".into()))?;

        let depth = DepthMap::from_raw(width, height, depth_buffer)?;

        Ok(RenderOutput {
            image,
            depth,
            alpha: alpha_buffer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraExtrinsics, CameraIntrinsics};
    use crate::gaussian::Gaussian3D;
    use glam::Quat;

    #[test]
    fn test_rasterizer_projection_and_render() {
        let mut cloud = GaussianCloud::new();
        cloud.gaussians.push(Gaussian3D::new(
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.2, 0.2, 0.2),
            Quat::IDENTITY,
            0.9,
            [1.0, 0.0, 0.0], // Red splat
        ));

        let camera = CameraPose::new(
            CameraExtrinsics::look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y),
            CameraIntrinsics::default_for_resolution(64, 64),
            0,
            0.0,
        );

        let rasterizer = GaussianRasterizer::new(RasterizerConfig::default());
        let projected = rasterizer.project_gaussians(&cloud, &camera);
        assert_eq!(projected.len(), 1);

        let out = rasterizer.render(&cloud, &camera).expect("Render should succeed");
        assert_eq!(out.image.width(), 64);
        assert_eq!(out.image.height(), 64);

        // Center pixel should be red
        let center_pixel = out.image.get_pixel(32, 32);
        assert!(center_pixel[0] > 100);
        assert!(out.alpha[32 * 64 + 32] > 0.5);
    }
}
