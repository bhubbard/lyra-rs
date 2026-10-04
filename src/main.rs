use clap::Parser;
use colored::Colorize;
use glam::Vec3;
use indicatif::{ProgressBar, ProgressStyle};
use lyra::camera::{CameraExtrinsics, CameraIntrinsics, CameraPose};
use lyra::cli::{Cli, Commands, GsCommands};
use lyra::depth::{DepthMap, PointCloud};
use lyra::error::Result;
use lyra::flow::{FlowSchedule, FnVelocityField, ODESolverType, RectifiedFlowSolver};
use lyra::gaussian::GaussianCloud;
use lyra::ply::{load_ply, save_ply};
use lyra::rasterizer::{GaussianRasterizer, RasterizerConfig};
use lyra::server::{create_router, ServerState};
use lyra::trajectory::{Trajectory, TrajectoryConfig, TrajectoryType};
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Traj(args) => {
            println!(
                "{} Generating camera trajectory...",
                "▶".bright_cyan().bold()
            );

            let traj_type = match args.traj_type.to_lowercase().as_str() {
                "spiral" => TrajectoryType::Spiral,
                "spiral_inward" => TrajectoryType::SpiralInward,
                "spiral_outward" => TrajectoryType::SpiralOutward,
                "orbit" | "orbit_horizontal" => TrajectoryType::OrbitHorizontal,
                "orbit_vertical" => TrajectoryType::OrbitVertical,
                "dolly" | "dolly_zoom" => TrajectoryType::DollyZoom,
                "zoom_in" => TrajectoryType::ZoomIn,
                "zoom_out" => TrajectoryType::ZoomOut,
                "zoom" | "zoom_in_out" => TrajectoryType::ZoomInOut,
                "horizontal" => TrajectoryType::Horizontal,
                "horizontal_lift" => TrajectoryType::HorizontalLift,
                "spline" => TrajectoryType::Spline,
                _ => TrajectoryType::OrbitHorizontal,
            };

            let config = TrajectoryConfig {
                trajectory_type: traj_type,
                num_frames: args.frames,
                fps: args.fps,
                strength: args.strength,
                target: Vec3::new(args.target[0], args.target[1], args.target[2]),
                initial_position: Vec3::new(args.eye[0], args.eye[1], args.eye[2]),
                initial_up: Vec3::Y,
                intrinsics: CameraIntrinsics::default_for_resolution(args.width, args.height),
                avoid_collision: true,
                collision_box: None,
            };

            let traj = Trajectory::generate(config)?;
            let json = traj.to_json()?;
            fs::write(&args.output, json)?;

            println!(
                "{} Generated {} frames ({:.1}s @ {} fps)",
                "✓".bright_green().bold(),
                traj.len().to_string().bright_yellow(),
                (traj.len() as f32) / (args.fps as f32),
                args.fps
            );
            println!(
                "  Trajectory saved to: {}",
                args.output.display().to_string().bright_white().bold()
            );
        }

        Commands::Gs(args) => match args.command {
            GsCommands::Render {
                input,
                output,
                eye,
                target,
                width,
                height,
            } => {
                println!(
                    "{} Preparing 3D Gaussian Splatting scene...",
                    "▶".bright_cyan().bold()
                );

                let cloud = if input == "demo" {
                    println!("  Using procedural courtyard 3DGS demo scene");
                    ServerState::demo().cloud.lock().unwrap().clone()
                } else {
                    println!("  Loading PLY file: {}", input.bright_white());
                    load_ply(&input)?
                };

                println!(
                    "{} Total Gaussians in scene: {}",
                    "✓".bright_green().bold(),
                    cloud.len().to_string().bright_yellow()
                );

                let extrinsics = CameraExtrinsics::look_at(
                    Vec3::new(eye[0], eye[1], eye[2]),
                    Vec3::new(target[0], target[1], target[2]),
                    Vec3::Y,
                );
                let intrinsics = CameraIntrinsics::default_for_resolution(width, height);
                let pose = CameraPose::new(extrinsics, intrinsics, 0, 0.0);

                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} {msg}")
                        .unwrap(),
                );
                pb.set_message("Rasterizing 3D Gaussians (pure Rust multi-threaded)...");

                let rasterizer = GaussianRasterizer::new(RasterizerConfig::default());
                let render_out = rasterizer.render(&cloud, &pose)?;

                pb.finish_and_clear();

                render_out.image.save(&output)?;
                println!(
                    "{} Rendered image saved to: {}",
                    "✓".bright_green().bold(),
                    output.display().to_string().bright_white().bold()
                );
            }

            GsCommands::Info { input } => {
                println!("{} Inspecting 3DGS PLY: {}", "▶".bright_cyan(), input.display());
                let cloud = load_ply(&input)?;
                let bbox = cloud.bounding_box();
                println!("  Gaussians count: {}", cloud.len().to_string().bright_yellow());
                println!(
                    "  Bounding Box: [{:.2}, {:.2}, {:.2}] to [{:.2}, {:.2}, {:.2}]",
                    bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z
                );
                println!(
                    "  Scene Extents: [{:.2}, {:.2}, {:.2}]",
                    bbox.size().x, bbox.size().y, bbox.size().z
                );
            }

            GsCommands::Demo { output } => {
                let cloud = ServerState::demo().cloud.lock().unwrap().clone();
                save_ply(&cloud, &output)?;
                println!(
                    "{} Saved demo scene ({} splats) to: {}",
                    "✓".bright_green().bold(),
                    cloud.len().to_string().bright_yellow(),
                    output.display().to_string().bright_white().bold()
                );
            }
        },

        Commands::Depth(args) => {
            println!(
                "{} Generating depth field and lifting to 3D point cloud...",
                "▶".bright_cyan().bold()
            );

            let mut depth_map = DepthMap::new(args.size, args.size, 2.0);
            for y in 0..args.size {
                for x in 0..args.size {
                    let fx = (x as f32) / (args.size as f32) - 0.5;
                    let fy = (y as f32) / (args.size as f32) - 0.5;
                    let r = (fx * fx + fy * fy).sqrt();
                    // Spherical bump
                    let d = 2.0 - (1.0 - (r * 2.0).clamp(0.0, 1.0)).powi(2) * 0.8;
                    depth_map.set(x, y, d);
                }
            }

            let camera = CameraPose::new(
                CameraExtrinsics::look_at(Vec3::new(0.0, 0.0, 3.5), Vec3::ZERO, Vec3::Y),
                CameraIntrinsics::default_for_resolution(args.size, args.size),
                0,
                0.0,
            );

            let pcd = PointCloud::from_rgbd(&depth_map, None, &camera, 0.1, 10.0, 1)?;
            println!("  Initial unprojected points: {}", pcd.len().to_string().bright_yellow());

            let pcd_filtered = if args.voxel_size > 0.0 {
                let down = pcd.voxel_downsample(args.voxel_size);
                println!(
                    "  After voxel downsampling ({:.3}m): {}",
                    args.voxel_size,
                    down.len().to_string().bright_yellow()
                );
                down
            } else {
                pcd
            };

            let cloud = GaussianCloud::from_point_cloud(&pcd_filtered, Some(args.voxel_size * 0.5))?;
            save_ply(&cloud, &args.output)?;

            println!(
                "{} Exported lifted 3DGS scene to: {}",
                "✓".bright_green().bold(),
                args.output.display().to_string().bright_white().bold()
            );
        }

        Commands::Flow(args) => {
            println!(
                "{} Simulating Rectified Flow Matching ODE...",
                "▶".bright_cyan().bold()
            );

            let solver_type = match args.solver.to_lowercase().as_str() {
                "euler" => ODESolverType::Euler,
                "midpoint" => ODESolverType::Midpoint,
                "rk4" => ODESolverType::Rk4,
                _ => ODESolverType::Heun,
            };

            let mut schedule = FlowSchedule::default();
            schedule.num_steps = args.steps;
            let sigmas = schedule.compute_sigmas();

            println!("  Solver: {:?}", solver_type);
            println!("  Steps:  {}", args.steps);
            println!("  Shift:  {}", schedule.shift);
            println!("  Sigmas: {:?}", &sigmas);

            // Synthetic velocity field
            let vf = FnVelocityField(|x, _t, out| {
                for i in 0..x.len() {
                    out[i] = -x[i] * 0.8;
                }
                Ok(())
            });

            let solver = RectifiedFlowSolver::new(solver_type, schedule);
            let initial = vec![1.0f32, -0.5f32, 2.0f32];
            let out = solver.solve(&initial, &vf)?;

            println!("  Initial sample (noise): {:?}", initial);
            println!("  Generated sample (x0):   {:?}", out);
            println!(
                "{} Flow ODE integration completed successfully!",
                "✓".bright_green().bold()
            );
        }

        Commands::Serve(args) => {
            println!(
                "{} Launching Project Lyra Interactive 3D World Streaming Server...",
                "🚀".bright_cyan().bold()
            );

            let server_state = if let Some(ply_path) = args.input_ply {
                println!("  Loading 3D Gaussians from: {}", ply_path.display());
                let cloud = load_ply(&ply_path)?;
                ServerState::new(cloud)
            } else {
                println!("  Starting with procedural courtyard 3DGS world");
                ServerState::demo()
            };

            let addr = format!("{}:{}", args.host, args.port);
            let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
                lyra::error::LyraError::ApiError(format!("Failed to bind to {}: {}", addr, e))
            })?;

            println!("{} Server listening on http://{}", "✓".bright_green().bold(), addr);
            println!("  Web Viewer:       http://{}/", addr);
            println!("  Health Endpoint:  http://{}/health", addr);
            println!("  World State API:  http://{}/api/v1/state", addr);
            println!("  WebSocket Stream: ws://{}/ws/stream", addr);
            println!("\nPress Ctrl+C to terminate server.\n");

            let router = create_router(server_state);
            axum::serve(listener, router).await.map_err(|e| {
                lyra::error::LyraError::ApiError(format!("Server runtime error: {}", e))
            })?;
        }
    }

    Ok(())
}
