# Lyra-RS: Pure Rust Generative 3D World Models & Gaussian Splatting

[![Crates.io](https://img.shields.io/badge/crates.io-v0.1.0-orange.svg)](https://crates.io)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Build & Tests](https://img.shields.io/badge/tests-21%2F21%20passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()

A high-performance, pure Rust engine and native fork of NVIDIA's **[Project Lyra](https://github.com/nv-tlabs/lyra)** (Open Generative 3D World Models & Gaussian Splatting).

`lyra-rs` implements the core mathematical, algorithmic, and serving capabilities of Lyra 1.0 and Lyra 2.0 without requiring Python, PyTorch, or proprietary CUDA toolchains. It runs cross-platform on Apple Silicon, Linux, and Windows with multi-threaded CPU acceleration.

---

## 🌟 Key Features

### 1. 3D Gaussian Splatting Software Rasterizer
- Pure Rust parallel CPU rasterizer powered by [`rayon`](https://crates.io/crates/rayon) and [`glam`](https://crates.io/crates/glam).
- Jacobian screen-space projection: converts 3D world covariance $\Sigma = R S S^T R^T$ into 2D screen covariance $\Sigma_{2D} = J W \Sigma W^T J^T$.
- View-dependent color via Spherical Harmonics ($Y_{lm}$) up to degree 3.
- Hierarchical tile-based binning ($16 \times 16$ pixel blocks) with front-to-back alpha compositing.
- Produces color image buffers, expected depth maps, and accumulation alpha masks.

### 2. Camera Geometry & Trajectory Generator
- Analytical generators for camera motion:
  - **Orbit**: Circular orbit around target with customizable radius, height, and elevation.
  - **Spiral**: Inward/outward spiral trajectory.
  - **Dolly-Zoom (Vertigo effect)**: Synchronized camera translation and dynamic FOV variation keeping foreground scale invariant.
  - **Zoom In / Out**: Linear longitudinal camera motion.
  - **Spline**: Smooth multi-waypoint interpolation via Catmull-Rom splines and quaternion SLERP.
- Plücker ray coordinates: $L = (d, o \times d)$ for ray embeddings.
- Full compatibility with Lyra's `trajectory.npz` camera extrinsics (C2W, W2C) and intrinsics matrices.

### 3. Geometry-Guided Information Routing
- RGB-D unprojection into metric 3D point clouds.
- 3D voxel grid downsampling with centroid computation.
- **Frustum overlap and visibility scoring**: Computes camera viewing frustum intersections to dynamically route historical keyframes and eliminate spatial forgetting during long-horizon exploration.

### 4. Rectified Flow Matching ODE Solvers
- Numerical ODE solvers: **Euler**, **Midpoint**, **Heun** (2nd-order Runge-Kutta), and **RK4** (4th-order Runge-Kutta).
- Shifted $\sigma$-noise schedule with resolution-dependent scaling $\sigma_t = \frac{s \cdot t}{1 + (s - 1) \cdot t}$.
- Lyra 2.0 DMD (Distribution Matching Distillation) 4-step accelerated trajectory schedule.
- **Autoregressive chunk rollout manager** with boundary tracking (81-frame chunks with 1-frame keyframe overlap) to prevent autoregressive drift.

### 5. Spatio-Temporal 3D Diffusion Transformer (DiT)
- 3D Rotary Position Embeddings (3D RoPE) decomposing frequencies across $(t, h, w)$.
- Scaled dot-product multi-head attention.
- Adaptive Layer Normalization with Zero-initialization (AdaLN-Zero) for condition-driven modulation.

### 6. Standard 3DGS PLY Serializer & Parser
- Full compatibility with standard 3D Gaussian Splatting `.ply` format (SuperSplat, Nerfstudio, WebGL viewers, CloudCompare).
- Supports both high-speed little-endian binary format and readable ASCII format.
- Handles spherical harmonics coefficients (`f_dc_0..2`, `f_rest_0..44`) and inverse sigmoid opacity mapping.

### 7. Interactive Streaming Server & HTML5 Web Viewer
- Built with [`axum`](https://crates.io/crates/axum) and [`tokio`](https://crates.io/crates/tokio).
- WebSocket binary streaming delivering live rendered Gaussian viewpoints with camera updates.
- REST API for world state inspection, trajectory generation, and single-frame rendering.
- Embedded zero-dependency HTML5 3D interactive viewer with orbit controls, FOV slider, and frame HUD.

---

## 🚀 Quick Start

### Installation

Ensure you have Rust 1.80+ installed.

```bash
git clone https://github.com/bhubbard/lyra-rs.git
cd lyra-rs
cargo build --release
```

The compiled binary will be located at `target/release/lyra`.

---

## 🛠️ CLI Usage

```bash
lyra --help
```

### 1. Camera Trajectory Generation (`traj`)

Generate camera poses and export them to JSON:

```bash
# Circular orbit (81 frames, 30 FPS, radius 4.0m)
lyra traj --traj-type orbit --frames 81 --fps 30.0 --radius 4.0 --height 1.5 --output orbit.json

# Cinematic dolly-zoom (vertigo effect)
lyra traj --traj-type dolly-zoom --frames 81 --output dolly.json

# Smooth spline walkthrough
lyra traj --traj-type spline --frames 120 --output spline.json
```

### 2. 3D Gaussian Splatting Tools (`gs`)

```bash
# Generate a synthetic demo Gaussian cloud (courtyard fountain)
lyra gs demo --output scene.ply

# Inspect any standard 3DGS PLY file
lyra gs info --input scene.ply

# Render a camera viewpoint to PNG
lyra gs render \
  --input scene.ply \
  --output render.png \
  --eye 0.0,2.0,5.0 \
  --target 0.0,0.0,0.0 \
  --width 1280 \
  --height 720 \
  --fov 60.0
```

### 3. RGB-D Lifting & Point Cloud Voxelization (`depth`)

```bash
# Generate a test depth map, lift to point cloud, and voxel downsample
lyra depth --size 128 --voxel-size 0.05 --output cloud.ply
```

### 4. Rectified Flow ODE Solver (`flow`)

```bash
# Run flow matching ODE with Heun solver and 4 distillation steps
lyra flow --solver heun --steps 4 --shift 3.0

# Run high-precision RK4 solver with 20 steps
lyra flow --solver rk4 --steps 20
```

### 5. Launch Interactive 3D World Server (`serve`)

```bash
# Start server on default port 8080 with built-in demo scene
lyra serve --port 8080

# Or serve a custom Gaussian PLY file
lyra serve --port 8080 --ply my_world.ply
```

Navigate to `http://localhost:8080` in your web browser to interact with the 3D world in real-time.

---

## 🌐 HTTP & WebSocket API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Embedded HTML5 interactive 3D world viewer |
| `/health` | `GET` | Service health check |
| `/api/v1/state` | `GET` | Returns active world metadata, Gaussian count, and bounding box |
| `/api/v1/generate` | `POST` | Generates a multi-frame camera trajectory chunk |
| `/api/v1/render` | `POST` | Renders a single viewpoint and returns a Base64 PNG |
| `/ws/stream` | `WS` | Real-time bidirectional WebSocket: client sends camera state, server streams rendered frames |

---

## 📦 Rust Library Usage

Add `lyra-rs` to your `Cargo.toml`:

```toml
[dependencies]
lyra = { path = "../lyra-rs" }
glam = "0.29"
```

### Example: Loading and Rendering Gaussians

```rust
use lyra::{
    CameraExtrinsics, CameraIntrinsics, CameraPose,
    GaussianCloud, GaussianRasterizer, RasterizerConfig,
};
use glam::Vec3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load Gaussian Cloud from standard PLY
    let cloud = GaussianCloud::load_ply("scene.ply")?;
    println!("Loaded {} Gaussians", cloud.len());

    // 2. Set up Camera
    let extrinsics = CameraExtrinsics::look_at(
        Vec3::new(0.0, 2.0, 4.0), // Eye
        Vec3::ZERO,                // Target
        Vec3::Y,                   // Up
    );
    let intrinsics = CameraIntrinsics::default_for_resolution(1280, 720);
    let camera = CameraPose::new(extrinsics, intrinsics, 0, 0.0);

    // 3. Render viewpoint with CPU multi-threading
    let rasterizer = GaussianRasterizer::new(RasterizerConfig::default());
    let output = rasterizer.render(&cloud, &camera)?;

    // 4. Save image
    output.image.save("output.png")?;
    println!("Rendered to output.png");

    Ok(())
}
```

---

## 🔬 Mathematical Architecture & Lyra Parity

| Component | NVIDIA Lyra (Python / CUDA) | Lyra-RS (Pure Rust) | Parity Status |
| :--- | :--- | :--- | :--- |
| **Gaussian Representation** | 3D Gaussian ($\mu, s, q, \alpha, \text{SH}$) | `Gaussian3D` with $Y_{lm}$ up to degree 3 | Full Parity |
| **Rasterization** | Custom CUDA Kernels (`diff-gaussian-rasterization`) | Pure Rust parallel Rayon CPU software rasterizer | Full Parity (CPU) |
| **PLY I/O** | `plyfile` Python wrapper | Native binary little-endian & ASCII reader/writer | Full Parity |
| **Trajectory Generation** | `trajectory.py` (orbit, spiral, dolly, spline) | `lyra::trajectory::Trajectory` | Full Parity |
| **Information Routing** | Point cloud unprojection & visibility overlap | `lyra::depth::route_informative_views` | Full Parity |
| **Flow Matching ODE** | Euler & RK4 ODE solvers with shifted schedule | `lyra::flow::RectifiedFlowSolver` (Euler, Midpoint, Heun, RK4) | Full Parity |
| **Autoregressive Rollout**| 81-frame chunks with 1-frame overlap | `lyra::flow::AutoregressiveChunkManager` | Full Parity |
| **Spatio-Temporal DiT** | Wan 2.1 3D DiT with 3D RoPE & AdaLN-Zero | `lyra::dit::{RoPE3D, Attention, AdaLNZero}` | Full Parity (Architectural) |
| **Interactive GUI** | C++ / CUDA ImGui Viewer | Axum HTTP/WebSocket Server + Zero-Dependency Web Viewer | Cross-platform alternative |

---

## 🧪 Testing

Run the full unit test suite:

```bash
cargo test
```

All 21 comprehensive test suites verify camera projections, Catmull-Rom splines, Plücker coordinates, Gaussian covariance math, tile rasterization, PLY round-tripping, ODE solver accuracy, and autoregressive chunk boundaries.

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
