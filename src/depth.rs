use crate::camera::CameraPose;
use crate::error::{LyraError, Result};
use crate::math::BoundingBox;
use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 2D floating-point depth map buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl DepthMap {
    pub fn new(width: u32, height: u32, initial_depth: f32) -> Self {
        Self {
            width,
            height,
            data: vec![initial_depth; (width * height) as usize],
        }
    }

    pub fn from_raw(width: u32, height: u32, data: Vec<f32>) -> Result<Self> {
        if data.len() != (width * height) as usize {
            return Err(LyraError::InvalidCamera(format!(
                "Depth map buffer length mismatch: expected {}, got {}",
                width * height,
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            0.0
        } else {
            self.data[(y * self.width + x) as usize]
        }
    }

    pub fn set(&mut self, x: u32, y: u32, val: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = val;
        }
    }

    pub fn min_max(&self) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &d in &self.data {
            if d.is_finite() && d > 0.0 {
                if d < min {
                    min = d;
                }
                if d > max {
                    max = d;
                }
            }
        }
        if min == f32::INFINITY {
            (0.0, 0.0)
        } else {
            (min, max)
        }
    }
}

/// A 3D colored point with optional normal vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub position: Vec3,
    pub color: [u8; 3],
    pub normal: Option<Vec3>,
}

/// A collection of 3D points forming a scene geometry representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud {
    pub points: Vec<Point3D>,
}

impl PointCloud {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn bounding_box(&self) -> BoundingBox {
        let mut bbox = BoundingBox::empty();
        for p in &self.points {
            bbox.include_point(p.position);
        }
        bbox
    }

    /// Lifts a 2D depth map and RGB image buffer into 3D world space points using the camera pose.
    pub fn from_rgbd(
        depth_map: &DepthMap,
        rgb_data: Option<&[u8]>,
        camera: &CameraPose,
        min_depth: f32,
        max_depth: f32,
        subsample_step: u32,
    ) -> Result<Self> {
        let w = depth_map.width;
        let h = depth_map.height;
        let step = subsample_step.max(1);

        let mut points = Vec::new();

        for y in (0..h).step_by(step as usize) {
            for x in (0..w).step_by(step as usize) {
                let depth = depth_map.get(x, y);

                if !depth.is_finite() || depth < min_depth || depth > max_depth {
                    continue;
                }

                if let Ok(pos) = camera.unproject_point(x as f32, y as f32, depth) {
                    let color = if let Some(rgb) = rgb_data {
                        let idx = ((y * w + x) * 3) as usize;
                        if idx + 2 < rgb.len() {
                            [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
                        } else {
                            [200, 200, 200]
                        }
                    } else {
                        // Colorize by normalized depth
                        let norm = ((depth - min_depth) / (max_depth - min_depth).max(1e-4)).clamp(0.0, 1.0);
                        let val = (norm * 255.0) as u8;
                        [val, val, 255 - val]
                    };

                    points.push(Point3D {
                        position: pos,
                        color,
                        normal: None,
                    });
                }
            }
        }

        Ok(Self { points })
    }

    /// Downsamples the point cloud using a 3D spatial voxel grid.
    pub fn voxel_downsample(&self, voxel_size: f32) -> Self {
        if voxel_size <= 0.0 || self.points.is_empty() {
            return self.clone();
        }

        use std::collections::HashMap;

        // Group points by 3D integer voxel coordinates (i, j, k)
        let mut grid: HashMap<(i32, i32, i32), Vec<Point3D>> = HashMap::new();

        for pt in &self.points {
            let i = (pt.position.x / voxel_size).floor() as i32;
            let j = (pt.position.y / voxel_size).floor() as i32;
            let k = (pt.position.z / voxel_size).floor() as i32;
            grid.entry((i, j, k)).or_default().push(*pt);
        }

        let mut downsampled = Vec::with_capacity(grid.len());
        for pts in grid.values() {
            let count = pts.len() as f32;
            let mut avg_pos = Vec3::ZERO;
            let mut avg_r = 0.0;
            let mut avg_g = 0.0;
            let mut avg_b = 0.0;

            for p in pts {
                avg_pos += p.position;
                avg_r += p.color[0] as f32;
                avg_g += p.color[1] as f32;
                avg_b += p.color[2] as f32;
            }

            downsampled.push(Point3D {
                position: avg_pos / count,
                color: [
                    (avg_r / count) as u8,
                    (avg_g / count) as u8,
                    (avg_b / count) as u8,
                ],
                normal: None,
            });
        }

        Self {
            points: downsampled,
        }
    }
}

/// Represents information routing score between a target query viewpoint and a historical view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewRoutingScore {
    pub history_frame_idx: usize,
    pub overlap_score: f32,
    pub angular_distance_deg: f32,
    pub camera_distance: f32,
}

/// Computes the visual/frustum overlap and routes relevant historical frames for long-horizon 3D consistency.
///
/// This addresses spatial forgetting in Lyra 2.0: retrieving past viewpoints that share geometry
/// with the current query pose.
pub fn route_informative_views(
    target_pose: &CameraPose,
    history_poses: &[CameraPose],
    scene_points: &PointCloud,
    top_k: usize,
) -> Vec<ViewRoutingScore> {
    if history_poses.is_empty() {
        return Vec::new();
    }

    let target_pos = target_pose.extrinsics.position();
    let target_forward = target_pose.extrinsics.forward();

    // Sample subset of points to check visibility from both views
    let sample_points: Vec<Vec3> = if scene_points.len() > 200 {
        let step = scene_points.len() / 200;
        scene_points
            .points
            .iter()
            .step_by(step)
            .map(|p| p.position)
            .collect()
    } else {
        scene_points.points.iter().map(|p| p.position).collect()
    };

    let mut scores: Vec<ViewRoutingScore> = history_poses
        .par_iter()
        .map(|hist_pose| {
            let hist_pos = hist_pose.extrinsics.position();
            let hist_forward = hist_pose.extrinsics.forward();

            let camera_dist = target_pos.distance(hist_pos);
            let dot = target_forward.dot(hist_forward).clamp(-1.0, 1.0);
            let angle_deg = dot.acos().to_degrees();

            // Estimate overlap by checking how many scene points project into both frustums
            let mut mutual_visible = 0usize;
            let mut target_visible = 0usize;

            for &pt in &sample_points {
                let in_target = target_pose.project_point(pt).map_or(false, |(uv, d)| {
                    d > 0.1
                        && uv.x >= 0.0
                        && uv.x < target_pose.intrinsics.width as f32
                        && uv.y >= 0.0
                        && uv.y < target_pose.intrinsics.height as f32
                });

                if in_target {
                    target_visible += 1;
                    let in_hist = hist_pose.project_point(pt).map_or(false, |(uv, d)| {
                        d > 0.1
                            && uv.x >= 0.0
                            && uv.x < hist_pose.intrinsics.width as f32
                            && uv.y >= 0.0
                            && uv.y < hist_pose.intrinsics.height as f32
                    });
                    if in_hist {
                        mutual_visible += 1;
                    }
                }
            }

            let overlap_ratio = if target_visible > 0 {
                (mutual_visible as f32) / (target_visible as f32)
            } else {
                // Fallback to geometric angle/distance proxy if no points visible
                ((-angle_deg / 45.0).exp() * (-camera_dist / 2.0).exp()).clamp(0.0, 1.0)
            };

            ViewRoutingScore {
                history_frame_idx: hist_pose.frame_idx,
                overlap_score: overlap_ratio,
                angular_distance_deg: angle_deg,
                camera_distance: camera_dist,
            }
        })
        .collect();

    // Sort descending by overlap score
    scores.sort_by(|a, b| {
        b.overlap_score
            .partial_cmp(&a.overlap_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scores.truncate(top_k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraExtrinsics, CameraIntrinsics};

    #[test]
    fn test_point_cloud_lifting_and_downsampling() {
        let mut depth_map = DepthMap::new(10, 10, 2.0);
        depth_map.set(5, 5, 3.0);

        let camera = CameraPose::new(
            CameraExtrinsics::look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y),
            CameraIntrinsics::default_for_resolution(10, 10),
            0,
            0.0,
        );

        let pcd = PointCloud::from_rgbd(&depth_map, None, &camera, 0.1, 10.0, 1)
            .expect("PCD lifting should succeed");
        assert_eq!(pcd.len(), 100);

        let downsampled = pcd.voxel_downsample(0.5);
        assert!(downsampled.len() < pcd.len());
        assert!(!downsampled.is_empty());
    }

    #[test]
    fn test_information_routing() {
        let intrinsics = CameraIntrinsics::default_for_resolution(64, 64);
        let pose_target = CameraPose::new(
            CameraExtrinsics::look_at(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y),
            intrinsics,
            10,
            1.0,
        );

        let pose_near = CameraPose::new(
            CameraExtrinsics::look_at(Vec3::new(0.1, 0.0, 3.0), Vec3::ZERO, Vec3::Y),
            intrinsics,
            0,
            0.0,
        );

        let pose_far = CameraPose::new(
            CameraExtrinsics::look_at(Vec3::new(10.0, 0.0, 3.0), Vec3::new(10.0, 0.0, 0.0), Vec3::Y),
            intrinsics,
            1,
            0.1,
        );

        let mut pcd = PointCloud::new();
        pcd.points.push(Point3D {
            position: Vec3::new(0.0, 0.0, 0.0),
            color: [255, 0, 0],
            normal: None,
        });

        let routed = route_informative_views(&pose_target, &[pose_near, pose_far], &pcd, 2);
        assert_eq!(routed.len(), 2);
        assert_eq!(routed[0].history_frame_idx, 0); // pose_near should have higher overlap
        assert!(routed[0].overlap_score >= routed[1].overlap_score);
    }
}
