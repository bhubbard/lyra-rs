use crate::camera::CameraPose;
use crate::trajectory::TrajectoryType;
use serde::{Deserialize, Serialize};

/// Status of an asynchronous chunk generation job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    GeneratingTrajectory,
    SolvingFlow,
    Rasterizing,
    Completed,
    Failed,
}

/// Request payload for generating an exploratory 3D world chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldGenerationRequest {
    pub request_id: String,
    pub trajectory_type: TrajectoryType,
    pub num_frames: usize,
    pub fps: u32,
    pub prompt: Option<String>,
    pub camera_target: [f32; 3],
    pub camera_position: [f32; 3],
    pub strength: f32,
    pub width: u32,
    pub height: u32,
}

impl Default for WorldGenerationRequest {
    fn default() -> Self {
        Self {
            request_id: "req_default".to_string(),
            trajectory_type: TrajectoryType::OrbitHorizontal,
            num_frames: 81,
            fps: 16,
            prompt: Some("A sunny photorealistic courtyard garden with stone pathways".to_string()),
            camera_target: [0.0, 0.0, 0.0],
            camera_position: [0.0, 1.0, 3.5],
            strength: 1.0,
            width: 640,
            height: 360,
        }
    }
}

/// Interactive camera pose update from the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCameraUpdate {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_deg: Option<f32>,
}

/// Streamed video or 3D frame chunk packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub request_id: String,
    pub frame_idx: usize,
    pub total_frames: usize,
    pub timestamp_sec: f32,
    pub camera_pose: CameraPose,
    pub status: JobStatus,
    /// Base64-encoded JPEG/PNG image data or URL.
    pub image_base64: Option<String>,
}

/// Current state and statistics of the explorable 3D world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub world_id: String,
    pub total_keyframes: usize,
    pub gaussian_splats_count: usize,
    pub bounding_box_min: [f32; 3],
    pub bounding_box_max: [f32; 3],
    pub active_viewers: usize,
}
