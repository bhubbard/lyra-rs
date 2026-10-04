//! Project Lyra (Rust Fork): Open Generative 3D World Models and Gaussian Splatting.
//!
//! Developed by NVIDIA Spatial Intelligence Lab; pure Rust implementation by Brandon Hubbard.

pub mod api;
pub mod camera;
pub mod cli;
pub mod depth;
pub mod dit;
pub mod error;
pub mod flow;
pub mod gaussian;
pub mod math;
pub mod ply;
pub mod rasterizer;
pub mod server;
pub mod trajectory;

pub use camera::{CameraExtrinsics, CameraIntrinsics, CameraPose};
pub use depth::{DepthMap, PointCloud};
pub use error::{LyraError, Result};
pub use flow::{FlowSchedule, ODESolverType, RectifiedFlowSolver};
pub use gaussian::{Gaussian3D, GaussianCloud};
pub use rasterizer::{GaussianRasterizer, RasterizerConfig, RenderOutput};
pub use server::ServerState;
pub use trajectory::{Trajectory, TrajectoryConfig, TrajectoryType};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
