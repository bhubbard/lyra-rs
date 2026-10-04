use serde::{Deserialize, Serialize};

/// 3D Spatio-Temporal Patch Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchConfig3D {
    pub patch_time: usize,
    pub patch_height: usize,
    pub patch_width: usize,
    pub in_channels: usize,
    pub hidden_dim: usize,
}

impl Default for PatchConfig3D {
    fn default() -> Self {
        Self {
            patch_time: 1,
            patch_height: 2,
            patch_width: 2,
            in_channels: 16, // Latent channels
            hidden_dim: 1536,
        }
    }
}

/// 3D Rotary Positional Embedding (RoPE) frequencies across Time, Height, Width.
pub struct RotaryEmbedding3D {
    pub dim: usize,
    pub theta: f32,
}

impl RotaryEmbedding3D {
    pub fn new(dim: usize, theta: f32) -> Self {
        Self { dim, theta }
    }

    /// Computes 3D RoPE cos and sin tables for coordinates (t, y, x).
    pub fn compute_frequencies(&self, t: usize, y: usize, x: usize) -> (Vec<f32>, Vec<f32>) {
        let dim_each = self.dim / 3;
        let mut cos_table = Vec::with_capacity(self.dim);
        let mut sin_table = Vec::with_capacity(self.dim);

        // Time component
        for i in 0..(dim_each / 2) {
            let freq = 1.0 / self.theta.powf((2 * i) as f32 / dim_each as f32);
            let val = (t as f32) * freq;
            cos_table.push(val.cos());
            sin_table.push(val.sin());
        }

        // Height (Y) component
        for i in 0..(dim_each / 2) {
            let freq = 1.0 / self.theta.powf((2 * i) as f32 / dim_each as f32);
            let val = (y as f32) * freq;
            cos_table.push(val.cos());
            sin_table.push(val.sin());
        }

        // Width (X) component
        for i in 0..(dim_each / 2) {
            let freq = 1.0 / self.theta.powf((2 * i) as f32 / dim_each as f32);
            let val = (x as f32) * freq;
            cos_table.push(val.cos());
            sin_table.push(val.sin());
        }

        (cos_table, sin_table)
    }

    /// Applies 2D rotary embedding to paired elements in an activation slice.
    pub fn apply_rotary(x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let n = cos.len().min(x.len() / 2);
        for i in 0..n {
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            let c = cos[i];
            let s = sin[i];
            x[2 * i] = x0 * c - x1 * s;
            x[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

/// Adaptive LayerNorm (AdaLN-Zero) Modulation Parameters.
///
/// Modulates normalized activations by: y = gate * (x * (1 + scale) + shift)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaLNModulation {
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
    pub gate: Vec<f32>,
}

impl AdaLNModulation {
    pub fn zero(dim: usize) -> Self {
        Self {
            shift: vec![0.0; dim],
            scale: vec![0.0; dim],
            gate: vec![0.0; dim],
        }
    }

    pub fn modulate(&self, x: &mut [f32]) {
        let dim = x.len().min(self.scale.len());
        for i in 0..dim {
            let val = x[i] * (1.0 + self.scale[i]) + self.shift[i];
            x[i] = val * self.gate[i];
        }
    }
}

/// Computes multi-head scaled dot-product attention in pure Rust.
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len_q: usize,
    seq_len_k: usize,
    dim: usize,
    causal: bool,
) {
    let scale = 1.0 / (dim as f32).sqrt();

    for i in 0..seq_len_q {
        let q_vec = &q[i * dim..(i + 1) * dim];

        // Compute unnormalized attention logits
        let mut max_logit = f32::NEG_INFINITY;
        let mut logits = vec![0.0f32; seq_len_k];

        let max_k_idx = if causal { i + 1 } else { seq_len_k };

        for j in 0..max_k_idx {
            let k_vec = &k[j * dim..(j + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q_vec[d] * k_vec[d];
            }
            let score = dot * scale;
            logits[j] = score;
            if score > max_logit {
                max_logit = score;
            }
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        let mut probs = vec![0.0f32; seq_len_k];
        for j in 0..max_k_idx {
            let exp_val = (logits[j] - max_logit).exp();
            probs[j] = exp_val;
            sum_exp += exp_val;
        }

        let inv_sum = if sum_exp > 1e-6 { 1.0 / sum_exp } else { 0.0 };
        for j in 0..max_k_idx {
            probs[j] *= inv_sum;
        }

        // Weighted value sum
        let out_slice = &mut out[i * dim..(i + 1) * dim];
        for d in 0..dim {
            out_slice[d] = 0.0;
        }

        for j in 0..max_k_idx {
            let v_vec = &v[j * dim..(j + 1) * dim];
            let p = probs[j];
            for d in 0..dim {
                out_slice[d] += p * v_vec[d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rotary_embedding_3d() {
        let rope = RotaryEmbedding3D::new(60, 10000.0);
        let (cos, sin) = rope.compute_frequencies(2, 5, 8);
        assert_eq!(cos.len(), 30);
        assert_eq!(sin.len(), 30);

        let mut x = vec![1.0, 0.0];
        RotaryEmbedding3D::apply_rotary(&mut x, &cos[0..1], &sin[0..1]);

        // Length should be preserved under rotation
        let len_sq = x[0] * x[0] + x[1] * x[1];
        assert_relative_eq!(len_sq, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];

        scaled_dot_product_attention(&q, &k, &v, &mut out, 2, 2, 2, false);

        assert!(out[0] > 0.0);
        assert!(out[1] > 0.0);
    }

    #[test]
    fn test_adaln_zero_modulation() {
        let mut mod_params = AdaLNModulation::zero(3);
        mod_params.scale = vec![0.5, 0.5, 0.5];
        mod_params.shift = vec![1.0, 1.0, 1.0];
        mod_params.gate = vec![1.0, 1.0, 1.0];

        let mut x = vec![2.0, 4.0, 6.0];
        mod_params.modulate(&mut x);

        assert_relative_eq!(x[0], 2.0 * 1.5 + 1.0); // 4.0
        assert_relative_eq!(x[1], 4.0 * 1.5 + 1.0); // 7.0
        assert_relative_eq!(x[2], 6.0 * 1.5 + 1.0); // 10.0
    }
}
