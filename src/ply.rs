use crate::error::{LyraError, Result};
use crate::gaussian::{Gaussian3D, GaussianCloud};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use glam::{Quat, Vec3};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Standard spherical harmonic DC factor: C0 = 0.28209479177387814 = 1 / (2 * sqrt(pi))
const SH_C0: f32 = 0.28209479177387814;

/// Saves a `GaussianCloud` to a standard binary little-endian 3DGS `.ply` file.
pub fn save_ply<P: AsRef<Path>>(cloud: &GaussianCloud, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let count = cloud.len();

    // Write PLY header
    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "element vertex {}", count)?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    writeln!(writer, "property float nx")?;
    writeln!(writer, "property float ny")?;
    writeln!(writer, "property float nz")?;
    writeln!(writer, "property float f_dc_0")?;
    writeln!(writer, "property float f_dc_1")?;
    writeln!(writer, "property float f_dc_2")?;
    writeln!(writer, "property float opacity")?;
    writeln!(writer, "property float scale_0")?;
    writeln!(writer, "property float scale_1")?;
    writeln!(writer, "property float scale_2")?;
    writeln!(writer, "property float rot_0")?;
    writeln!(writer, "property float rot_1")?;
    writeln!(writer, "property float rot_2")?;
    writeln!(writer, "property float rot_3")?;
    writeln!(writer, "end_header")?;

    for g in &cloud.gaussians {
        // x, y, z
        writer.write_f32::<LittleEndian>(g.position.x)?;
        writer.write_f32::<LittleEndian>(g.position.y)?;
        writer.write_f32::<LittleEndian>(g.position.z)?;

        // nx, ny, nz (normals, usually 0)
        writer.write_f32::<LittleEndian>(0.0)?;
        writer.write_f32::<LittleEndian>(0.0)?;
        writer.write_f32::<LittleEndian>(0.0)?;

        // f_dc_0, f_dc_1, f_dc_2: (rgb - 0.5) / SH_C0
        let f_dc_0 = (g.color_rgb[0] - 0.5) / SH_C0;
        let f_dc_1 = (g.color_rgb[1] - 0.5) / SH_C0;
        let f_dc_2 = (g.color_rgb[2] - 0.5) / SH_C0;
        writer.write_f32::<LittleEndian>(f_dc_0)?;
        writer.write_f32::<LittleEndian>(f_dc_1)?;
        writer.write_f32::<LittleEndian>(f_dc_2)?;

        // Inverse sigmoid opacity: logit(alpha) = ln(alpha / (1 - alpha))
        let alpha = g.opacity.clamp(1e-4, 0.9999);
        let logit_alpha = (alpha / (1.0 - alpha)).ln();
        writer.write_f32::<LittleEndian>(logit_alpha)?;

        // Log-scales
        let s0 = (g.scale.x.max(1e-6)).ln();
        let s1 = (g.scale.y.max(1e-6)).ln();
        let s2 = (g.scale.z.max(1e-6)).ln();
        writer.write_f32::<LittleEndian>(s0)?;
        writer.write_f32::<LittleEndian>(s1)?;
        writer.write_f32::<LittleEndian>(s2)?;

        // Rotation quaternion: (w, x, y, z)
        writer.write_f32::<LittleEndian>(g.rotation.w)?;
        writer.write_f32::<LittleEndian>(g.rotation.x)?;
        writer.write_f32::<LittleEndian>(g.rotation.y)?;
        writer.write_f32::<LittleEndian>(g.rotation.z)?;
    }

    writer.flush()?;
    Ok(())
}

/// Loads a `GaussianCloud` from a 3DGS `.ply` file (binary or ASCII).
pub fn load_ply<P: AsRef<Path>>(path: P) -> Result<GaussianCloud> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(LyraError::PlyError("Missing 'ply' magic header".into()));
    }

    let mut is_binary = false;
    let mut vertex_count = 0usize;
    let mut properties = Vec::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            return Err(LyraError::PlyError("Unexpected EOF before 'end_header'".into()));
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "format" => {
                if parts.len() > 1 && parts[1] == "binary_little_endian" {
                    is_binary = true;
                }
            }
            "element" => {
                if parts.len() > 2 && parts[1] == "vertex" {
                    vertex_count = parts[2].parse().unwrap_or(0);
                }
            }
            "property" => {
                if parts.len() > 2 {
                    properties.push((parts[1].to_string(), parts[2].to_string()));
                }
            }
            "end_header" => break,
            _ => {}
        }
    }

    if vertex_count == 0 {
        return Ok(GaussianCloud::new());
    }

    let mut gaussians = Vec::with_capacity(vertex_count);

    if is_binary {
        let prop_count = properties.len();

        let mut x_idx = None;
        let mut y_idx = None;
        let mut z_idx = None;
        let mut f_dc_0_idx = None;
        let mut f_dc_1_idx = None;
        let mut f_dc_2_idx = None;
        let mut opacity_idx = None;
        let mut scale_0_idx = None;
        let mut scale_1_idx = None;
        let mut scale_2_idx = None;
        let mut rot_0_idx = None;
        let mut rot_1_idx = None;
        let mut rot_2_idx = None;
        let mut rot_3_idx = None;

        for (idx, (_, name)) in properties.iter().enumerate() {
            match name.as_str() {
                "x" => x_idx = Some(idx),
                "y" => y_idx = Some(idx),
                "z" => z_idx = Some(idx),
                "f_dc_0" => f_dc_0_idx = Some(idx),
                "f_dc_1" => f_dc_1_idx = Some(idx),
                "f_dc_2" => f_dc_2_idx = Some(idx),
                "opacity" => opacity_idx = Some(idx),
                "scale_0" => scale_0_idx = Some(idx),
                "scale_1" => scale_1_idx = Some(idx),
                "scale_2" => scale_2_idx = Some(idx),
                "rot_0" => rot_0_idx = Some(idx),
                "rot_1" => rot_1_idx = Some(idx),
                "rot_2" => rot_2_idx = Some(idx),
                "rot_3" => rot_3_idx = Some(idx),
                _ => {}
            }
        }

        let mut float_buf = vec![0.0f32; prop_count];

        for _ in 0..vertex_count {
            for i in 0..prop_count {
                float_buf[i] = reader.read_f32::<LittleEndian>()?;
            }

            let pos = Vec3::new(
                x_idx.map_or(0.0, |i| float_buf[i]),
                y_idx.map_or(0.0, |i| float_buf[i]),
                z_idx.map_or(0.0, |i| float_buf[i]),
            );

            let r = f_dc_0_idx.map_or(0.5, |i| float_buf[i] * SH_C0 + 0.5).clamp(0.0, 1.0);
            let g = f_dc_1_idx.map_or(0.5, |i| float_buf[i] * SH_C0 + 0.5).clamp(0.0, 1.0);
            let b = f_dc_2_idx.map_or(0.5, |i| float_buf[i] * SH_C0 + 0.5).clamp(0.0, 1.0);

            // Sigmoid: 1 / (1 + exp(-x))
            let opacity = opacity_idx
                .map_or(1.0, |i| 1.0 / (1.0 + (-float_buf[i]).exp()))
                .clamp(0.0, 1.0);

            let scale = Vec3::new(
                scale_0_idx.map_or(0.01, |i| float_buf[i].exp()),
                scale_1_idx.map_or(0.01, |i| float_buf[i].exp()),
                scale_2_idx.map_or(0.01, |i| float_buf[i].exp()),
            );

            let rotation = Quat::from_xyzw(
                rot_1_idx.map_or(0.0, |i| float_buf[i]),
                rot_2_idx.map_or(0.0, |i| float_buf[i]),
                rot_3_idx.map_or(0.0, |i| float_buf[i]),
                rot_0_idx.map_or(1.0, |i| float_buf[i]),
            )
            .normalize();

            gaussians.push(Gaussian3D::new(pos, scale, rotation, opacity, [r, g, b]));
        }
    } else {
        // Fallback for ASCII PLY
        for _ in 0..vertex_count {
            line.clear();
            reader.read_line(&mut line)?;
            let vals: Vec<f32> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if vals.len() >= 3 {
                let pos = Vec3::new(vals[0], vals[1], vals[2]);
                gaussians.push(Gaussian3D::new(
                    pos,
                    Vec3::splat(0.02),
                    Quat::IDENTITY,
                    0.9,
                    [0.8, 0.8, 0.8],
                ));
            }
        }
    }

    Ok(GaussianCloud { gaussians })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ply_save_and_load_roundtrip() {
        let mut cloud = GaussianCloud::new();
        cloud.gaussians.push(Gaussian3D::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.05, 0.05, 0.05),
            Quat::IDENTITY,
            0.85,
            [0.2, 0.4, 0.6],
        ));

        let temp_dir = tempfile::tempdir().unwrap();
        let ply_path = temp_dir.path().join("test_splat.ply");

        save_ply(&cloud, &ply_path).expect("Save PLY should succeed");
        let loaded = load_ply(&ply_path).expect("Load PLY should succeed");

        assert_eq!(loaded.len(), 1);
        let g = &loaded.gaussians[0];
        assert_relative_eq!(g.position.x, 1.0, epsilon = 1e-4);
        assert_relative_eq!(g.position.y, 2.0, epsilon = 1e-4);
        assert_relative_eq!(g.position.z, 3.0, epsilon = 1e-4);
        assert_relative_eq!(g.opacity, 0.85, epsilon = 1e-2);
        assert_relative_eq!(g.color_rgb[0], 0.2, epsilon = 1e-2);
    }
}
