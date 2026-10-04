use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Type of numerical ODE solver for rectified flow matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ODESolverType {
    Euler,
    Midpoint,
    Heun,
    Rk4,
}

/// Timestep schedule configuration for rectified flow sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowSchedule {
    pub num_steps: usize,
    pub shift: f32,
    pub sigma_min: f32,
    pub sigma_max: f32,
    pub reverse: bool,
}

impl Default for FlowSchedule {
    fn default() -> Self {
        Self {
            num_steps: 20,
            shift: 3.0, // Lyra 2.0 default shift parameter
            sigma_min: 0.003 / 1.002,
            sigma_max: 1.0,
            reverse: false,
        }
    }
}

impl FlowSchedule {
    /// 4-step fast DMD distillation schedule configuration.
    pub fn dmd_distillation() -> Self {
        Self {
            num_steps: 4,
            shift: 3.0,
            sigma_min: 0.003 / 1.002,
            sigma_max: 1.0,
            reverse: false,
        }
    }

    /// Computes the discrete sigmas sequence $[\sigma_0, \sigma_1, \dots, \sigma_N]$
    /// applying the shift transformation $\sigma' = \frac{s \cdot \sigma}{1 + (s - 1) \cdot \sigma}$.
    pub fn compute_sigmas(&self) -> Vec<f32> {
        let n = self.num_steps;
        if n == 0 {
            return Vec::new();
        }

        let mut sigmas = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = (i as f32) / (n as f32);
            // Linear interpolation from sigma_max down to sigma_min
            let raw_sigma = self.sigma_max * (1.0 - t) + self.sigma_min * t;
            // Shift transformation
            let shifted = (self.shift * raw_sigma) / (1.0 + (self.shift - 1.0) * raw_sigma);
            sigmas.push(shifted);
        }

        if self.reverse {
            sigmas.reverse();
        }

        sigmas
    }
}

/// Velocity field trait representing the learned neural network drift v_theta(x, t, context).
pub trait VelocityField {
    fn evaluate(&self, x: &[f32], t: f32, output: &mut [f32]) -> Result<()>;
}

/// Closure-based velocity field helper.
pub struct FnVelocityField<F>(pub F)
where
    F: Fn(&[f32], f32, &mut [f32]) -> Result<()>;

impl<F> VelocityField for FnVelocityField<F>
where
    F: Fn(&[f32], f32, &mut [f32]) -> Result<()>,
{
    fn evaluate(&self, x: &[f32], t: f32, output: &mut [f32]) -> Result<()> {
        (self.0)(x, t, output)
    }
}

/// Rectified flow matching ODE solver.
pub struct RectifiedFlowSolver {
    pub solver_type: ODESolverType,
    pub schedule: FlowSchedule,
}

impl RectifiedFlowSolver {
    pub fn new(solver_type: ODESolverType, schedule: FlowSchedule) -> Self {
        Self {
            solver_type,
            schedule,
        }
    }

    /// Solves the flow matching ODE forward from t=1 (noise) to t=0 (clean sample):
    /// dx / dt = v(x, t)
    pub fn solve<V: VelocityField>(&self, initial_state: &[f32], velocity_field: &V) -> Result<Vec<f32>> {
        let sigmas = self.schedule.compute_sigmas();
        if sigmas.len() < 2 {
            return Ok(initial_state.to_vec());
        }

        let dim = initial_state.len();
        let mut x = initial_state.to_vec();

        let mut k1 = vec![0.0f32; dim];
        let mut k2 = vec![0.0f32; dim];
        let mut k3 = vec![0.0f32; dim];
        let mut k4 = vec![0.0f32; dim];
        let mut temp_state = vec![0.0f32; dim];

        for i in 0..(sigmas.len() - 1) {
            let t_curr = sigmas[i];
            let t_next = sigmas[i + 1];
            let dt = t_next - t_curr; // negative for denoising

            match self.solver_type {
                ODESolverType::Euler => {
                    velocity_field.evaluate(&x, t_curr, &mut k1)?;
                    for d in 0..dim {
                        x[d] += dt * k1[d];
                    }
                }
                ODESolverType::Midpoint => {
                    velocity_field.evaluate(&x, t_curr, &mut k1)?;
                    for d in 0..dim {
                        temp_state[d] = x[d] + 0.5 * dt * k1[d];
                    }
                    velocity_field.evaluate(&temp_state, t_curr + 0.5 * dt, &mut k2)?;
                    for d in 0..dim {
                        x[d] += dt * k2[d];
                    }
                }
                ODESolverType::Heun => {
                    // RK2 / Heun's method
                    velocity_field.evaluate(&x, t_curr, &mut k1)?;
                    for d in 0..dim {
                        temp_state[d] = x[d] + dt * k1[d];
                    }
                    velocity_field.evaluate(&temp_state, t_next, &mut k2)?;
                    for d in 0..dim {
                        x[d] += 0.5 * dt * (k1[d] + k2[d]);
                    }
                }
                ODESolverType::Rk4 => {
                    velocity_field.evaluate(&x, t_curr, &mut k1)?;

                    for d in 0..dim {
                        temp_state[d] = x[d] + 0.5 * dt * k1[d];
                    }
                    velocity_field.evaluate(&temp_state, t_curr + 0.5 * dt, &mut k2)?;

                    for d in 0..dim {
                        temp_state[d] = x[d] + 0.5 * dt * k2[d];
                    }
                    velocity_field.evaluate(&temp_state, t_curr + 0.5 * dt, &mut k3)?;

                    for d in 0..dim {
                        temp_state[d] = x[d] + dt * k3[d];
                    }
                    velocity_field.evaluate(&temp_state, t_next, &mut k4)?;

                    for d in 0..dim {
                        x[d] += (dt / 6.0) * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]);
                    }
                }
            }
        }

        Ok(x)
    }
}

/// Autoregressive Chunking & Self-Forcing Rollout Manager.
///
/// Manages transitions across chunks (e.g. 81, 161, 241 frames) and maintains
/// self-augmented histories to prevent temporal drift in long-horizon world exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoregressiveChunkManager {
    pub chunk_size: usize,
    pub overlap_frames: usize,
    pub total_frames_generated: usize,
    pub current_chunk_idx: usize,
}

impl AutoregressiveChunkManager {
    pub fn new(chunk_size: usize, overlap_frames: usize) -> Self {
        Self {
            chunk_size,
            overlap_frames,
            total_frames_generated: 0,
            current_chunk_idx: 0,
        }
    }

    /// Standard Lyra 2.0 configuration (81 frames per chunk with 1 overlap frame).
    pub fn lyra2_standard() -> Self {
        Self::new(81, 1)
    }

    /// Checks if a frame index represents a chunk boundary.
    pub fn is_chunk_boundary(&self, frame_idx: usize) -> bool {
        if frame_idx == 0 {
            return false;
        }
        let stride = self.chunk_size.saturating_sub(self.overlap_frames);
        if stride == 0 {
            return false;
        }
        frame_idx % stride == 0
    }

    /// Step the chunk manager forward by one chunk.
    pub fn advance_chunk(&mut self) {
        self.current_chunk_idx += 1;
        self.total_frames_generated += self.chunk_size - self.overlap_frames;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_flow_schedule_sigmas() {
        let schedule = FlowSchedule::dmd_distillation();
        let sigmas = schedule.compute_sigmas();

        assert_eq!(sigmas.len(), 5); // 4 steps -> 5 points
        assert!(sigmas[0] > sigmas[4]); // Decreasing noise level
        assert_relative_eq!(sigmas[0], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ode_solvers_on_linear_flow() {
        // v(x, t) = -x, exact solution: x(t) = x_0 * exp(-(1 - t))
        let vf = FnVelocityField(|x, _t, out| {
            for i in 0..x.len() {
                out[i] = -x[i];
            }
            Ok(())
        });

        let mut sched = FlowSchedule::default();
        sched.num_steps = 20;

        let solver_euler = RectifiedFlowSolver::new(ODESolverType::Euler, sched.clone());
        let solver_heun = RectifiedFlowSolver::new(ODESolverType::Heun, sched.clone());
        let solver_rk4 = RectifiedFlowSolver::new(ODESolverType::Rk4, sched);

        let initial = vec![1.0f32, 2.0f32];
        let out_euler = solver_euler.solve(&initial, &vf).unwrap();
        let out_heun = solver_heun.solve(&initial, &vf).unwrap();
        let out_rk4 = solver_rk4.solve(&initial, &vf).unwrap();

        // Higher order solvers should be more accurate
        assert!(out_rk4[0] > 0.0);
        assert!((out_rk4[0] - out_heun[0]).abs() < 0.1);
        assert!((out_heun[0] - out_euler[0]).abs() < 0.2);
    }

    #[test]
    fn test_autoregressive_chunk_boundaries() {
        let mgr = AutoregressiveChunkManager::lyra2_standard();
        // 81 frames, 1 overlap -> chunk boundaries at 80, 160, 240
        assert!(mgr.is_chunk_boundary(80));
        assert!(mgr.is_chunk_boundary(160));
        assert!(mgr.is_chunk_boundary(240));
        assert!(!mgr.is_chunk_boundary(40));
    }
}
