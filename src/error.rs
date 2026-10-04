use thiserror::Error;

#[derive(Error, Debug)]
pub enum LyraError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization/JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Invalid camera parameters: {0}")]
    InvalidCamera(String),

    #[error("Invalid trajectory: {0}")]
    InvalidTrajectory(String),

    #[error("PLY format error: {0}")]
    PlyError(String),

    #[error("Gaussian splatting error: {0}")]
    GaussianError(String),

    #[error("ODE flow solver error: {0}")]
    FlowError(String),

    #[error("API / Server error: {0}")]
    ApiError(String),
}

pub type Result<T> = std::result::Result<T, LyraError>;
