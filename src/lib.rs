//! # Candlelight: Unified Candle ML Framework Wrapper
//!
//! This crate provides a single, unified dependency for the entire Candle ML framework ecosystem,
//! ensuring version compatibility across all Candle components.
//!
//! ## Why Candlelight?
//!
//! The Candle ecosystem consists of multiple crates that must be version-synchronized:
//! - `candle-core` - Core tensor operations
//! - `candle-nn` - Neural network layers
//! - `candle-transformers` - Transformer model implementations
//! - `candle-flash-attn` - FlashAttention-2 CUDA optimization
//! - `candle-layer-norm` - Fused LayerNorm/RMSNorm kernels
//!
//! Managing these dependencies across multiple projects is error-prone. Candlelight solves this
//! by providing a single dependency that guarantees all Candle components work together.
//!
//! ## Current Status
//!
//! **Using Git Snapshot**: This crate currently depends on a Candle git snapshot
//! (rev `db08cc0a5a786e00f873c35ced7db51fd7d7083a`) to support:
//! - CUDA 13.0 via cudarc 0.17.1+
//! - Compatibility with Visual Studio 2022 v17.12+
//!
//! Once Candle v0.10 is released, we'll switch to stable crates.io releases.
//!
//! ## Features
//!
//! ### Hardware Acceleration
//! - `cuda` - Enable CUDA GPU support (NVIDIA GPUs)
//! - `metal` - Enable Metal GPU support (Apple Silicon)
//!
//! ### CUDA Optimizations (require `cuda`)
//! - `flash-attn` - FlashAttention-2 for efficient attention computation
//! - `layer-norm` - Fused LayerNorm/RMSNorm kernels (~20-30% speedup)
//! - `cuda-full` - Convenience feature enabling all CUDA optimizations
//!
//! ### Utilities
//! - `datasets` - Dataset loading utilities
//!
//! ## Usage
//!
//! ```toml
//! [dependencies]
//! candlelight = { path = "../candlelight", features = ["cuda", "flash-attn"] }
//! ```
//!
//! ```rust,no_run
//! use candlelight::{Device, Tensor, Result};
//!
//! fn main() -> Result<()> {
//!     let device = Device::cuda_if_available(0)?;
//!     let tensor = Tensor::randn(0f32, 1.0, (2, 3), &device)?;
//!     println!("{tensor}");
//!     Ok(())
//! }
//! ```
//!
//! ## Re-exports
//!
//! All Candle crates are re-exported under intuitive module names:
//! - `candlelight::core` → `candle_core`
//! - `candlelight::nn` → `candle_nn`
//! - `candlelight::transformers` → `candle_transformers`
//! - `candlelight::flash_attn` → `candle_flash_attn` (when `flash-attn` feature enabled)
//! - `candlelight::layer_norm` → `candle_layer_norm` (when `layer-norm` feature enabled)
//! - `candlelight::datasets` → `candle_datasets` (when `datasets` feature enabled)
//!
//! Common types are also re-exported at the crate root for convenience.

// Re-export Candle crates
pub use candle_core as core;
pub use candle_nn as nn;
pub use candle_transformers as transformers;

#[cfg(feature = "flash-attn")]
pub use candle_flash_attn as flash_attn;

#[cfg(feature = "layer-norm")]
pub use candle_layer_norm as layer_norm;

#[cfg(feature = "datasets")]
pub use candle_datasets as datasets;

// Re-export commonly used types for convenience
pub use candle_core::{DType, Device, Error, Result, Shape, Tensor};
pub use candle_nn::VarBuilder;

/// Module containing all Candle core functionality
pub mod prelude {
    pub use candle_core::{DType, Device, Error, Module, Result, Shape, Tensor};
    pub use candle_nn::{
        loss, ops, Activation, Conv1d, Conv2d, Embedding, LayerNorm, Linear, RmsNorm, VarBuilder,
        VarMap,
    };

    #[cfg(feature = "flash-attn")]
    pub use candle_flash_attn::flash_attn;

    #[cfg(feature = "layer-norm")]
    pub use candle_layer_norm::{layer_norm, rms_norm};
}
