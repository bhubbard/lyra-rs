use crate::api::{ClientCameraUpdate, StreamChunk, WorldGenerationRequest, WorldState};
use crate::camera::{CameraExtrinsics, CameraIntrinsics, CameraPose};
use crate::gaussian::{Gaussian3D, GaussianCloud};
use crate::rasterizer::{GaussianRasterizer, RasterizerConfig};
use crate::trajectory::{Trajectory, TrajectoryConfig};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use glam::{Quat, Vec3};
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

/// Shared application state for the Lyra server.
#[derive(Clone)]
pub struct ServerState {
    pub cloud: Arc<Mutex<GaussianCloud>>,
    pub world_id: String,
    pub rasterizer: Arc<GaussianRasterizer>,
}

impl ServerState {
    pub fn new(cloud: GaussianCloud) -> Self {
        Self {
            cloud: Arc::new(Mutex::new(cloud)),
            world_id: "lyra-world-001".to_string(),
            rasterizer: Arc::new(GaussianRasterizer::new(RasterizerConfig::default())),
        }
    }

    /// Constructs a demo scene with colored 3D Gaussians arranged in a courtyard.
    pub fn demo() -> Self {
        let mut cloud = GaussianCloud::new();

        // Ground plane splats
        for x in -5..=5 {
            for z in -5..=5 {
                let px = (x as f32) * 0.6;
                let pz = (z as f32) * 0.6;
                let color = if (x + z) % 2 == 0 {
                    [0.4, 0.45, 0.4]
                } else {
                    [0.6, 0.65, 0.6]
                };
                cloud.gaussians.push(Gaussian3D::new(
                    Vec3::new(px, -0.5, pz),
                    Vec3::new(0.35, 0.05, 0.35),
                    Quat::IDENTITY,
                    0.95,
                    color,
                ));
            }
        }

        // Center feature statue / fountain
        for i in 0..12 {
            let angle = (i as f32) / 12.0 * std::f32::consts::TAU;
            let radius = 1.0;
            let px = radius * angle.cos();
            let pz = radius * angle.sin();
            let py = 0.5 + 0.3 * (angle * 2.0).sin();

            cloud.gaussians.push(Gaussian3D::new(
                Vec3::new(px, py, pz),
                Vec3::splat(0.2),
                Quat::IDENTITY,
                0.9,
                [0.2, 0.6, 0.9], // Blue fountain rim
            ));
        }

        // Center pillar
        for h in 0..5 {
            let py = (h as f32) * 0.4;
            cloud.gaussians.push(Gaussian3D::new(
                Vec3::new(0.0, py, 0.0),
                Vec3::new(0.2, 0.25, 0.2),
                Quat::IDENTITY,
                0.95,
                [0.9, 0.8, 0.3], // Golden center
            ));
        }

        Self::new(cloud)
    }
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/health", get(health_handler))
        .route("/api/v1/state", get(state_handler))
        .route("/api/v1/generate", post(generate_handler))
        .route("/api/v1/render", post(render_handler))
        .route("/ws/stream", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "OK - Lyra 3D World Model Server Active"
}

async fn state_handler(State(state): State<ServerState>) -> Json<WorldState> {
    let cloud = state.cloud.lock().unwrap();
    let bbox = cloud.bounding_box();
    Json(WorldState {
        world_id: state.world_id.clone(),
        total_keyframes: 81,
        gaussian_splats_count: cloud.len(),
        bounding_box_min: [bbox.min.x, bbox.min.y, bbox.min.z],
        bounding_box_max: [bbox.max.x, bbox.max.y, bbox.max.z],
        active_viewers: 1,
    })
}

async fn generate_handler(
    State(_state): State<ServerState>,
    Json(req): Json<WorldGenerationRequest>,
) -> Result<Json<Vec<StreamChunk>>, StatusCode> {
    let mut config = TrajectoryConfig::default();
    config.trajectory_type = req.trajectory_type;
    config.num_frames = req.num_frames;
    config.fps = req.fps;
    config.strength = req.strength;
    config.target = Vec3::from_array(req.camera_target);
    config.initial_position = Vec3::from_array(req.camera_position);
    config.intrinsics = CameraIntrinsics::default_for_resolution(req.width, req.height);

    let traj = Trajectory::generate(config)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut chunks = Vec::with_capacity(traj.len());
    for pose in &traj.poses {
        chunks.push(StreamChunk {
            request_id: req.request_id.clone(),
            frame_idx: pose.frame_idx,
            total_frames: traj.len(),
            timestamp_sec: pose.timestamp,
            camera_pose: pose.clone(),
            status: crate::api::JobStatus::Completed,
            image_base64: None,
        });
    }

    Ok(Json(chunks))
}

async fn render_handler(
    State(state): State<ServerState>,
    Json(cam): Json<ClientCameraUpdate>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let eye = Vec3::from_array(cam.eye);
    let target = Vec3::from_array(cam.target);
    let up = Vec3::from_array(cam.up);

    let extrinsics = CameraExtrinsics::look_at(eye, target, up);
    let fov = cam.fov_deg.unwrap_or(60.0).to_radians();
    let intrinsics = CameraIntrinsics::from_fov(fov, fov * 0.75, 640, 360);
    let pose = CameraPose::new(extrinsics, intrinsics, 0, 0.0);

    let cloud = state.cloud.lock().unwrap();
    let render_out = state
        .rasterizer
        .render(&cloud, &pose)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Encode to PNG bytes
    let mut png_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut png_bytes);
    render_out
        .image
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    use image::EncodableLayout;
    let b64 = format!(
        "data:image/png;base64,{}",
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, png_bytes.as_bytes())
    );

    Ok(Json(serde_json::json!({
        "status": "success",
        "image_data": b64,
        "width": 640,
        "height": 360,
    })))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: ServerState) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            if let Ok(cam_update) = serde_json::from_str::<ClientCameraUpdate>(&text) {
                let eye = Vec3::from_array(cam_update.eye);
                let target = Vec3::from_array(cam_update.target);
                let up = Vec3::from_array(cam_update.up);

                let extrinsics = CameraExtrinsics::look_at(eye, target, up);
                let fov = cam_update.fov_deg.unwrap_or(60.0).to_radians();
                let intrinsics = CameraIntrinsics::from_fov(fov, fov * 0.75, 480, 270);
                let pose = CameraPose::new(extrinsics, intrinsics, 0, 0.0);

                let render_result = {
                    let cloud = state.cloud.lock().unwrap();
                    state.rasterizer.render(&cloud, &pose)
                };
                if let Ok(render_out) = render_result {
                    let mut jpeg_bytes = Vec::new();
                    let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
                    let _ = render_out
                        .image
                        .write_to(&mut cursor, image::ImageFormat::Jpeg);

                    let _ = socket.send(Message::Binary(jpeg_bytes.into())).await;
                }
            }
        }
    }
}

async fn index_handler() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Project Lyra: Interactive 3D World Model Viewer</title>
  <style>
    body { margin: 0; background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; height: 100vh; }
    header { background: #161b22; padding: 12px 20px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
    h1 { margin: 0; font-size: 16px; font-weight: 600; color: #58a6ff; display: flex; align-items: center; gap: 8px; }
    .badge { background: #238636; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
    main { flex: 1; display: flex; overflow: hidden; }
    #viewport-container { flex: 1; position: relative; background: #010409; display: flex; justify-content: center; align-items: center; }
    #viewport { max-width: 100%; max-height: 100%; border: 1px solid #30363d; border-radius: 6px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
    aside { width: 320px; background: #161b22; border-left: 1px solid #30363d; padding: 16px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
    .card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
    .card h3 { margin-top: 0; font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
    button { background: #238636; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: background 0.15s; width: 100%; }
    button:hover { background: #2ea043; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .controls button { background: #21262d; border: 1px solid #30363d; font-size: 12px; }
    .controls button:hover { background: #30363d; }
    .stat-row { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px; }
  </style>
</head>
<body>
  <header>
    <h1>Project Lyra 3D World Model <span class="badge">Rust Engine</span></h1>
    <span style="font-size: 12px; color: #8b949e;">Interactive Real-time Exploration</span>
  </header>
  <main>
    <div id="viewport-container">
      <img id="viewport" src="" alt="Lyra 3D World View">
    </div>
    <aside>
      <div class="card">
        <h3>Camera Trajectory</h3>
        <div class="controls">
          <button onclick="orbit()">Orbit 360°</button>
          <button onclick="spiral()">Spiral</button>
          <button onclick="zoom()">Zoom Walk</button>
          <button onclick="resetCam()">Reset</button>
        </div>
      </div>
      <div class="card">
        <h3>World Status</h3>
        <div class="stat-row"><span>Architecture:</span><strong>Wan 2.1 DiT + 3DGS</strong></div>
        <div class="stat-row"><span>Rasterizer:</span><strong>Pure Rust Multi-threaded</strong></div>
        <div class="stat-row"><span>Scheduler:</span><strong>Rectified Flow (DMD)</strong></div>
        <div class="stat-row"><span>Gaussians:</span><strong id="stat-splats">-</strong></div>
      </div>
    </aside>
  </main>
  <script>
    let eye = [0.0, 1.5, 4.0];
    const target = [0.0, 0.0, 0.0];
    const up = [0.0, 1.0, 0.0];

    async function updateFrame() {
      try {
        const res = await fetch('/api/v1/render', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ eye, target, up, fov_deg: 60.0 })
        });
        const data = await res.json();
        if (data.image_data) {
          document.getElementById('viewport').src = data.image_data;
        }
      } catch (err) {
        console.error(err);
      }
    }

    async function loadStats() {
      const res = await fetch('/api/v1/state');
      const data = await res.json();
      document.getElementById('stat-splats').textContent = data.gaussian_splats_count;
    }

    function resetCam() {
      eye = [0.0, 1.5, 4.0];
      updateFrame();
    }

    let animId = null;
    function orbit() {
      if (animId) clearInterval(animId);
      let angle = 0;
      const radius = 4.0;
      animId = setInterval(() => {
        angle += 0.05;
        eye[0] = radius * Math.sin(angle);
        eye[2] = radius * Math.cos(angle);
        updateFrame();
      }, 50);
    }

    function spiral() {
      if (animId) clearInterval(animId);
      let angle = 0;
      let r = 4.0;
      animId = setInterval(() => {
        angle += 0.05;
        r = 2.0 + Math.sin(angle * 0.5) * 1.5;
        eye[0] = r * Math.sin(angle);
        eye[1] = 1.0 + Math.cos(angle * 0.5) * 0.8;
        eye[2] = r * Math.cos(angle);
        updateFrame();
      }, 50);
    }

    function zoom() {
      if (animId) clearInterval(animId);
      let t = 0;
      animId = setInterval(() => {
        t += 0.05;
        const d = 2.0 + Math.sin(t) * 1.8;
        eye = [0.0, 0.8, d];
        updateFrame();
      }, 50);
    }

    loadStats();
    updateFrame();
  </script>
</body>
</html>"#)
}
