use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "lyra",
    author = "Brandon Hubbard <bkhubbard@gmail.com>, NVIDIA Spatial Intelligence Lab (Lyra)",
    version,
    about = "Pure Rust fork and high-performance engine for Project Lyra: Open Generative 3D World Models and Gaussian Splatting"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generate camera trajectories (orbit, spiral, zoom-in/out, spline, dolly zoom).
    Traj(TrajArgs),

    /// 3D Gaussian Splatting operations (rasterization, PLY conversion, inspection).
    Gs(GsArgs),

    /// Depth map backprojection and point cloud lifting.
    Depth(DepthArgs),

    /// Rectified Flow matching ODE solver simulation and testing.
    Flow(FlowArgs),

    /// Launch the interactive 3D world exploration streaming server.
    Serve(ServeArgs),
}

#[derive(Args, Debug)]
pub struct TrajArgs {
    /// Trajectory type: orbit, spiral, zoom_in, zoom_out, zoom_in_out, dolly_zoom, spline, horizontal
    #[arg(short, long, default_value = "orbit_horizontal")]
    pub traj_type: String,

    /// Number of frames to generate (e.g. 81 for 1 Lyra AR chunk)
    #[arg(short, long, default_value_t = 81)]
    pub frames: usize,

    /// Frames per second
    #[arg(long, default_value_t = 16)]
    pub fps: u32,

    /// Trajectory motion strength scale
    #[arg(short, long, default_value_t = 1.0)]
    pub strength: f32,

    /// Camera target point [X, Y, Z]
    #[arg(long, num_args = 3, default_values_t = [0.0, 0.0, 0.0])]
    pub target: Vec<f32>,

    /// Initial camera position [X, Y, Z]
    #[arg(long, num_args = 3, default_values_t = [0.0, 1.0, 3.5])]
    pub eye: Vec<f32>,

    /// Image resolution width
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Image resolution height
    #[arg(long, default_value_t = 720)]
    pub height: u32,

    /// Output trajectory JSON path
    #[arg(short, long, default_value = "trajectory.json")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct GsArgs {
    #[command(subcommand)]
    pub command: GsCommands,
}

#[derive(Subcommand, Debug)]
pub enum GsCommands {
    /// Render 3D Gaussians from a camera viewpoint to an image file.
    Render {
        /// Input 3DGS PLY file path (or 'demo' for procedural courtyard scene)
        #[arg(short, long, default_value = "demo")]
        input: String,

        /// Output rendered image path (e.g. render.png)
        #[arg(short, long, default_value = "render.png")]
        output: PathBuf,

        /// Camera eye position [X, Y, Z]
        #[arg(long, num_args = 3, default_values_t = [0.0, 1.5, 4.0])]
        eye: Vec<f32>,

        /// Camera target [X, Y, Z]
        #[arg(long, num_args = 3, default_values_t = [0.0, 0.0, 0.0])]
        target: Vec<f32>,

        /// Resolution width
        #[arg(long, default_value_t = 640)]
        width: u32,

        /// Resolution height
        #[arg(long, default_value_t = 360)]
        height: u32,
    },

    /// Inspect a 3D Gaussian Splatting PLY file.
    Info {
        /// Path to the .ply file
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Generate procedural sample 3D Gaussians and export to standard PLY.
    Demo {
        /// Output .ply file path
        #[arg(short, long, default_value = "demo_scene.ply")]
        output: PathBuf,
    },
}

#[derive(Args, Debug)]
pub struct DepthArgs {
    /// Generate a test depth map and backproject into 3D point cloud.
    #[arg(short, long, default_value_t = 128)]
    pub size: u32,

    /// Voxel downsample grid size (0.0 to disable)
    #[arg(long, default_value_t = 0.05)]
    pub voxel_size: f32,

    /// Output 3DGS PLY file
    #[arg(short, long, default_value = "lifted_pcd.ply")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct FlowArgs {
    /// ODE solver algorithm: euler, midpoint, heun, rk4
    #[arg(short, long, default_value = "heun")]
    pub solver: String,

    /// Number of integration steps (e.g. 4 for DMD, 20 for standard)
    #[arg(short, long, default_value_t = 4)]
    pub steps: usize,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,

    /// Host IP address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Optional 3DGS PLY file to serve (or uses procedural demo courtyard if omitted)
    #[arg(short, long)]
    pub input_ply: Option<PathBuf>,
}
