use crate::camera::{CameraExtrinsics, CameraIntrinsics, CameraPose};
use crate::error::{LyraError, Result};
use crate::math::{catmull_rom_spline, BoundingBox};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Type of preset trajectory supported by Lyra.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryType {
    Spiral,
    SpiralInward,
    SpiralOutward,
    OrbitHorizontal,
    OrbitVertical,
    DollyZoom,
    ZoomIn,
    ZoomOut,
    ZoomInOut,
    Horizontal,
    HorizontalLift,
    Spline,
}

/// Trajectory configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryConfig {
    pub trajectory_type: TrajectoryType,
    pub num_frames: usize,
    pub fps: u32,
    pub strength: f32,
    pub target: Vec3,
    pub initial_position: Vec3,
    pub initial_up: Vec3,
    pub intrinsics: CameraIntrinsics,
    pub avoid_collision: bool,
    pub collision_box: Option<BoundingBox>,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            trajectory_type: TrajectoryType::OrbitHorizontal,
            num_frames: 81, // 81 frames = 1 Lyra AR chunk
            fps: 16,
            strength: 1.0,
            target: Vec3::ZERO,
            initial_position: Vec3::new(0.0, 0.0, 4.0),
            initial_up: Vec3::Y,
            intrinsics: CameraIntrinsics::default_for_resolution(1280, 720),
            avoid_collision: true,
            collision_box: None,
        }
    }
}

/// Full camera trajectory sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub poses: Vec<CameraPose>,
    pub config: TrajectoryConfig,
}

impl Trajectory {
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Generates a camera trajectory based on the specified configuration.
    pub fn generate(config: TrajectoryConfig) -> Result<Self> {
        if config.num_frames == 0 {
            return Err(LyraError::InvalidTrajectory(
                "num_frames must be greater than 0".to_string(),
            ));
        }

        let poses = match config.trajectory_type {
            TrajectoryType::OrbitHorizontal => generate_orbit(&config, true),
            TrajectoryType::OrbitVertical => generate_orbit(&config, false),
            TrajectoryType::Spiral => generate_spiral(&config, 0.0),
            TrajectoryType::SpiralInward => generate_spiral(&config, -1.0),
            TrajectoryType::SpiralOutward => generate_spiral(&config, 1.0),
            TrajectoryType::DollyZoom => generate_dolly_zoom(&config),
            TrajectoryType::ZoomIn => generate_zoom(&config, true, false),
            TrajectoryType::ZoomOut => generate_zoom(&config, false, true),
            TrajectoryType::ZoomInOut => generate_zoom(&config, true, true),
            TrajectoryType::Horizontal => generate_horizontal(&config, false),
            TrajectoryType::HorizontalLift => generate_horizontal(&config, true),
            TrajectoryType::Spline => {
                let waypoints = vec![
                    config.initial_position,
                    config.initial_position + Vec3::new(1.0, 0.5, -0.5) * config.strength,
                    config.initial_position + Vec3::new(0.0, 1.0, -1.5) * config.strength,
                    config.target + Vec3::new(-1.0, 0.2, 2.0) * config.strength,
                ];
                generate_spline(&config, &waypoints)
            }
        };

        Ok(Self { poses, config })
    }

    /// Exports camera poses as flat C2W arrays [N, 4, 4] for interoperability.
    pub fn c2w_matrices(&self) -> Vec<[f32; 16]> {
        self.poses
            .iter()
            .map(|p| p.extrinsics.c2w.to_cols_array())
            .collect()
    }

    /// Exports camera poses as flat W2C arrays [N, 4, 4] (matching Lyra's trajectory.npz w2c).
    pub fn w2c_matrices(&self) -> Vec<[f32; 16]> {
        self.poses
            .iter()
            .map(|p| p.extrinsics.w2c.to_cols_array())
            .collect()
    }

    /// Exports camera intrinsics matrices [N, 3, 3] (matching Lyra's trajectory.npz intrinsics).
    pub fn intrinsics_matrices(&self) -> Vec<[f32; 9]> {
        self.poses
            .iter()
            .map(|p| p.intrinsics.to_mat3().to_cols_array())
            .collect()
    }

    /// Serialize trajectory to JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(LyraError::from)
    }

    /// Deserialize trajectory from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(LyraError::from)
    }
}

fn generate_orbit(config: &TrajectoryConfig, horizontal: bool) -> Vec<CameraPose> {
    let mut poses = Vec::with_capacity(config.num_frames);
    let radius = (config.initial_position - config.target).length() * config.strength;
    let up = config.initial_up.normalize();

    for i in 0..config.num_frames {
        let t = (i as f32) / (config.num_frames as f32);
        let angle = t * std::f32::consts::TAU;

        let pos = if horizontal {
            let x = config.target.x + radius * angle.cos();
            let z = config.target.z + radius * angle.sin();
            let y = config.initial_position.y;
            Vec3::new(x, y, z)
        } else {
            let y = config.target.y + radius * angle.sin();
            let z = config.target.z + radius * angle.cos();
            let x = config.initial_position.x;
            Vec3::new(x, y, z)
        };

        let extrinsics = CameraExtrinsics::look_at(pos, config.target, up);
        let timestamp = (i as f32) / (config.fps as f32);
        poses.push(CameraPose::new(extrinsics, config.intrinsics, i, timestamp));
    }

    poses
}

fn generate_spiral(config: &TrajectoryConfig, radius_bias: f32) -> Vec<CameraPose> {
    let mut poses = Vec::with_capacity(config.num_frames);
    let base_radius = (config.initial_position - config.target).length() * config.strength;
    let up = config.initial_up.normalize();
    let num_revolutions = 2.0_f32;

    for i in 0..config.num_frames {
        let t = (i as f32) / (config.num_frames.max(1) as f32);
        let angle = t * num_revolutions * std::f32::consts::TAU;

        // Radius adjustment: inward (shrinking), outward (expanding), or centered
        let r = if radius_bias < 0.0 {
            base_radius * (1.0 - 0.5 * t)
        } else if radius_bias > 0.0 {
            base_radius * (0.5 + 0.5 * t)
        } else {
            base_radius * (1.0 + 0.2 * (angle * 0.5).sin())
        };

        let y = config.initial_position.y + (t - 0.5) * config.strength;
        let x = config.target.x + r * angle.cos();
        let z = config.target.z + r * angle.sin();
        let pos = Vec3::new(x, y, z);

        let extrinsics = CameraExtrinsics::look_at(pos, config.target, up);
        let timestamp = (i as f32) / (config.fps as f32);
        poses.push(CameraPose::new(extrinsics, config.intrinsics, i, timestamp));
    }

    poses
}

fn generate_dolly_zoom(config: &TrajectoryConfig) -> Vec<CameraPose> {
    let mut poses = Vec::with_capacity(config.num_frames);
    let initial_dist = (config.initial_position - config.target).length();
    let forward_dir = (config.target - config.initial_position).normalize();
    let up = config.initial_up.normalize();

    for i in 0..config.num_frames {
        let t = (i as f32) / (config.num_frames.max(1) as f32);
        // Move camera closer: distance shrinks from initial_dist to 0.4 * initial_dist
        let current_dist = initial_dist * (1.0 - 0.6 * t * config.strength).max(0.1);
        let pos = config.target - forward_dir * current_dist;

        // Adjust focal length proportionally to keep target object the same visual scale:
        // scale = current_dist / initial_dist
        let fov_scale = current_dist / initial_dist;
        let mut intrinsics = config.intrinsics;
        intrinsics.fx = (config.intrinsics.fx * fov_scale).max(50.0);
        intrinsics.fy = (config.intrinsics.fy * fov_scale).max(50.0);

        let extrinsics = CameraExtrinsics::look_at(pos, config.target, up);
        let timestamp = (i as f32) / (config.fps as f32);
        poses.push(CameraPose::new(extrinsics, intrinsics, i, timestamp));
    }

    poses
}

fn generate_zoom(config: &TrajectoryConfig, zoom_in: bool, zoom_out: bool) -> Vec<CameraPose> {
    let mut poses = Vec::with_capacity(config.num_frames);
    let initial_pos = config.initial_position;
    let forward = (config.target - initial_pos).normalize();
    let total_dist = (config.target - initial_pos).length() * 0.75 * config.strength;
    let up = config.initial_up.normalize();

    let half = config.num_frames / 2;

    for i in 0..config.num_frames {
        let t = (i as f32) / (config.num_frames.max(1) as f32);

        let progress = if zoom_in && zoom_out {
            if i <= half {
                // Zooming in
                (i as f32) / (half as f32)
            } else {
                // Zooming out
                1.0 - ((i - half) as f32) / ((config.num_frames - half) as f32)
            }
        } else if zoom_in {
            t
        } else {
            1.0 - t
        };

        // Smooth cubic ease-in-out
        let smooth_t = progress * progress * (3.0 - 2.0 * progress);
        let mut pos = initial_pos + forward * (total_dist * smooth_t);

        // Optional collision avoidance against scene box
        if config.avoid_collision {
            if let Some(bbox) = &config.collision_box {
                if bbox.contains(pos) {
                    pos = initial_pos + forward * (total_dist * smooth_t * 0.8);
                }
            }
        }

        let extrinsics = CameraExtrinsics::look_at(pos, config.target, up);
        let timestamp = (i as f32) / (config.fps as f32);
        poses.push(CameraPose::new(extrinsics, config.intrinsics, i, timestamp));
    }

    poses
}

fn generate_horizontal(config: &TrajectoryConfig, with_lift: bool) -> Vec<CameraPose> {
    let mut poses = Vec::with_capacity(config.num_frames);
    let initial_pos = config.initial_position;
    let forward = (config.target - initial_pos).normalize();
    let right = forward.cross(config.initial_up).normalize();
    let up = config.initial_up.normalize();

    let span = 2.0 * config.strength;

    for i in 0..config.num_frames {
        let t = (i as f32) / (config.num_frames.max(1) as f32);
        let offset = (t - 0.5) * span;

        let mut pos = initial_pos + right * offset;
        if with_lift {
            pos.y += (t * std::f32::consts::PI).sin() * 0.5 * config.strength;
        }

        let extrinsics = CameraExtrinsics::look_at(pos, config.target, up);
        let timestamp = (i as f32) / (config.fps as f32);
        poses.push(CameraPose::new(extrinsics, config.intrinsics, i, timestamp));
    }

    poses
}

fn generate_spline(config: &TrajectoryConfig, waypoints: &[Vec3]) -> Vec<CameraPose> {
    if waypoints.len() < 2 {
        return generate_orbit(config, true);
    }

    let mut poses = Vec::with_capacity(config.num_frames);
    let up = config.initial_up.normalize();

    // Extend ends to satisfy 4-point Catmull-Rom
    let mut pts = Vec::with_capacity(waypoints.len() + 2);
    pts.push(waypoints[0] + (waypoints[0] - waypoints[1]));
    pts.extend_from_slice(waypoints);
    pts.push(waypoints[waypoints.len() - 1] + (waypoints[waypoints.len() - 1] - waypoints[waypoints.len() - 2]));

    let segments = pts.len() - 3;

    for i in 0..config.num_frames {
        let global_t = (i as f32) / (config.num_frames.max(1) as f32) * (segments as f32);
        let seg_idx = (global_t.floor() as usize).min(segments - 1);
        let local_t = global_t - (seg_idx as f32);

        let p0 = pts[seg_idx];
        let p1 = pts[seg_idx + 1];
        let p2 = pts[seg_idx + 2];
        let p3 = pts[seg_idx + 3];

        let pos = catmull_rom_spline(p0, p1, p2, p3, local_t);
        let extrinsics = CameraExtrinsics::look_at(pos, config.target, up);
        let timestamp = (i as f32) / (config.fps as f32);
        poses.push(CameraPose::new(extrinsics, config.intrinsics, i, timestamp));
    }

    poses
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trajectory_generation() {
        let mut config = TrajectoryConfig::default();
        config.num_frames = 81; // Standard Lyra chunk
        config.trajectory_type = TrajectoryType::OrbitHorizontal;

        let traj = Trajectory::generate(config).expect("Trajectory should generate");
        assert_eq!(traj.len(), 81);

        // First frame should be near initial pos
        let first_pose = &traj.poses[0];
        let last_pose = &traj.poses[80];

        assert_relative_eq!(first_pose.timestamp, 0.0);
        assert_relative_eq!(last_pose.timestamp, 80.0 / 16.0);

        let c2w_mats = traj.c2w_matrices();
        assert_eq!(c2w_mats.len(), 81);
        assert_eq!(c2w_mats[0].len(), 16);
    }

    #[test]
    fn test_dolly_zoom_fov_variation() {
        let mut config = TrajectoryConfig::default();
        config.num_frames = 10;
        config.trajectory_type = TrajectoryType::DollyZoom;

        let traj = Trajectory::generate(config).expect("Dolly zoom should generate");
        let initial_fx = traj.poses[0].intrinsics.fx;
        let final_fx = traj.poses[9].intrinsics.fx;

        // In dolly zoom, focal length changes
        assert!(final_fx < initial_fx);
    }

    #[test]
    fn test_json_roundtrip() {
        let config = TrajectoryConfig::default();
        let traj = Trajectory::generate(config).expect("Generate should succeed");
        let json = traj.to_json().expect("To json should succeed");
        let parsed = Trajectory::from_json(&json).expect("From json should succeed");

        assert_eq!(traj.len(), parsed.len());
        assert_eq!(traj.config.num_frames, parsed.config.num_frames);
    }
}
