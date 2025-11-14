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
//! **"It Just Works" Philosophy**: All features are enabled by default for maximum convenience.
//! Users who need minimal compilation times can opt-out with `default-features = false`.
//!
//! ### Hardware Acceleration
//! - `cuda` - Enable CUDA GPU support (NVIDIA GPUs)
//! - `metal` - Enable Metal GPU support (Apple Silicon)
//!
//! ### CUDA Optimizations (require `cuda`)
//! - `flash-attn` - FlashAttention-2 for efficient attention computation (default)
//! - `layer-norm` - Fused LayerNorm/RMSNorm kernels (~20-30% speedup) (default)
//! - `cuda-full` - Convenience feature enabling all CUDA optimizations
//!
//! ### Utilities (all enabled by default)
//! - `datasets` - Dataset loading utilities
//! - `optimizers` - Advanced gradient-based optimizers (Adam, SGD, RMSprop, etc.)
//! - `basin-hopping` - Global optimization via basin hopping
//!
//! ## Usage
//!
//! ### Default (Everything Enabled)
//! ```toml
//! [dependencies]
//! candlelight = { path = "../candlelight", features = ["cuda"] }
//! ```
//!
//! ### Minimal Build (Opt-out of defaults)
//! ```toml
//! [dependencies]
//! candlelight = { path = "../candlelight", default-features = false }
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
//! **Comprehensive Access**: Candlelight re-exports the entire Candle ecosystem for maximum convenience.
//! Since Rust eliminates dead code, unused re-exports add no runtime overhead.
//!
//! ### Core Access Patterns
//! ```rust,no_run
//! // Direct access to everything
//! use candlelight::*;
//!
//! // Specific module access
//! use candlelight::nn::*;
//! use candlelight::transformers_models::*;
//!
//! // Convenient prelude
//! use candlelight::prelude::*;
//! ```
//!
//! ### Module Structure
//! - `candlelight::*` → All of `candle_core` + common items
//! - `candlelight::nn::*` → Complete `candle_nn` module  
//! - `candlelight::transformers_models::*` → Complete `candle_transformers`
//! - `candlelight::backprop::*` → Complete `candle_core::backprop`
//! - `candlelight::quantized::*` → Complete `candle_core::quantized`
//! - `candlelight::safetensors::*` → Complete `candle_core::safetensors`
//! - `candlelight::flash_attention::*` → Complete `candle_flash_attn`
//! - `candlelight::fused_ops::*` → Complete `candle_layer_norm`
//! - `candlelight::data::*` → Complete `candle_datasets`
//! - `candlelight::optimizers::*` → Complete `candle_optimizers`
//! - `candlelight::basin_hopping::*` → Basin hopping global optimization
//! - `candlelight::prelude::*` → Curated selection of most common items

// Re-export Candle crates
pub use candle_core as core;
pub use candle_transformers as transformers;

#[cfg(feature = "flash-attn")]
pub use candle_flash_attn as flash_attn;

#[cfg(feature = "layer-norm")]
pub use candle_layer_norm as layer_norm;

#[cfg(feature = "datasets")]
pub use candle_datasets as datasets;

#[cfg(feature = "optimizers")]
pub use candle_optimisers as optimisers;

#[cfg(feature = "basin-hopping")]
pub use candle_bhop as bhop;

// =============================================================================
// Optional feature module re-exports
// =============================================================================

#[cfg(feature = "flash-attn")]
pub mod flash_attention {
    pub use candle_flash_attn::*;
}

#[cfg(feature = "layer-norm")]
pub mod fused_ops {
    pub use candle_layer_norm::*;
}

#[cfg(feature = "datasets")]
pub mod data {
    // Re-export the entire candle_datasets module (top-level items)
    pub use candle_datasets::*;

    // Submodules
    pub mod batcher {
        pub use candle_datasets::batcher::*;
    }

    pub mod hub {
        pub use candle_datasets::hub::*;
    }

    pub mod nlp {
        pub use candle_datasets::nlp::*;
    }

    pub mod vision {
        pub use candle_datasets::vision::*;
    }
}

#[cfg(feature = "optimizers")]
pub mod optimizers {
    // Re-export all top-level items (traits, enums, etc.)
    pub use candle_optimisers::*;

    // Re-export all optimizer modules
    pub use candle_optimisers::{
        adadelta, adagrad, adam, adamax, esgd, lbfgs, nadam, radam, rmsprop,
    };

    // Re-export optimizer structs at module level for convenience
    pub use candle_optimisers::adadelta::{Adadelta, ParamsAdaDelta};
    pub use candle_optimisers::adagrad::{Adagrad, ParamsAdaGrad};
    pub use candle_optimisers::adam::{Adam, ParamsAdam};
    pub use candle_optimisers::adamax::{Adamax, ParamsAdaMax};
    pub use candle_optimisers::esgd::{ParamsSGD, SGD};
    pub use candle_optimisers::lbfgs::{Lbfgs, ParamsLBFGS};
    pub use candle_optimisers::nadam::{NAdam, ParamsNAdam};
    pub use candle_optimisers::radam::{ParamsRAdam, RAdam};
    pub use candle_optimisers::rmsprop::{ParamsRMSprop, RMSprop};

    // Convenience aliases for common alternate names
    pub use esgd as sgd;
    pub use ParamsSGD as ParamsSgd;
    pub use RAdam as Radam;
    pub use RMSprop as Rmsprop;
    pub use SGD as Sgd;
}

#[cfg(feature = "basin-hopping")]
pub mod basin_hopping {
    // Re-export everything from candle-bhop
    pub use candle_bhop::*;
}

// =============================================================================
// candle_transformers module re-exports
// =============================================================================

pub mod transformers_models {
    // Re-export the entire candle_transformers module (top-level items)
    pub use candle_transformers::*;

    // Submodules
    pub mod generation {
        pub use candle_transformers::generation::*;
    }

    pub mod models {
        pub use candle_transformers::models::*;
    }

    pub mod object_detection {
        pub use candle_transformers::object_detection::*;
    }

    pub mod pipelines {
        pub use candle_transformers::pipelines::*;
    }

    pub mod quantized_nn {
        pub use candle_transformers::quantized_nn::*;
    }

    pub mod quantized_var_builder {
        pub use candle_transformers::quantized_var_builder::*;
    }

    pub mod utils {
        pub use candle_transformers::utils::*;
    }
}

// Re-export commonly used types for convenience from the root
pub use candle_core::{DType, Device, Error, IndexOp, Module, Result, Shape, Tensor, Var};
pub use candle_nn::VarBuilder;

// Re-export everything from candle-core at the root for maximum convenience
pub use candle_core::*;

// =============================================================================
// candle_nn module re-exports
// =============================================================================

pub mod nn {
    // Re-export the entire candle_nn module (top-level items)
    pub use candle_nn::*;

    // Submodules
    pub mod activation {
        pub use candle_nn::activation::*;
    }

    pub mod batch_norm {
        pub use candle_nn::batch_norm::*;
    }

    pub mod conv {
        pub use candle_nn::conv::*;
    }

    pub mod embedding {
        pub use candle_nn::embedding::*;
    }

    pub mod encoding {
        pub use candle_nn::encoding::*;
    }

    pub mod func {
        pub use candle_nn::func::*;
    }

    pub mod group_norm {
        pub use candle_nn::group_norm::*;
    }

    pub mod init {
        pub use candle_nn::init::*;
    }

    pub mod layer_norm {
        pub use candle_nn::layer_norm::*;
    }

    pub mod linear {
        pub use candle_nn::linear::*;
    }

    pub mod loss {
        pub use candle_nn::loss::*;
    }

    pub mod ops {
        pub use candle_nn::ops::*;
    }

    pub mod optim {
        pub use candle_nn::optim::*;
    }

    pub mod rnn {
        pub use candle_nn::rnn::*;
    }

    pub mod sequential {
        pub use candle_nn::sequential::*;
    }

    pub mod var_builder {
        pub use candle_nn::var_builder::*;
    }

    pub mod var_map {
        pub use candle_nn::var_map::*;
    }
}

// =============================================================================
// candle_core module re-exports
// =============================================================================

pub mod backend {
    pub use candle_core::backend::*;
}

pub mod backprop {
    pub use candle_core::backprop::*;
}

pub mod cpu {
    pub use candle_core::cpu::*;
}

pub mod cpu_backend {
    pub use candle_core::cpu_backend::*;
}

#[cfg(feature = "cuda")]
pub mod cuda_backend {
    pub use candle_core::cuda_backend::*;
}

#[cfg(feature = "cudnn")]
pub mod cudnn {
    pub use candle_core::cudnn::*;
}

pub mod display {
    pub use candle_core::display::*;
}

pub mod error {
    pub use candle_core::error::*;
}

pub mod layout {
    pub use candle_core::layout::*;
}

#[cfg(feature = "metal")]
pub mod metal_backend {
    pub use candle_core::metal_backend::*;
}

pub mod npy {
    pub use candle_core::npy::*;
}

pub mod pickle {
    pub use candle_core::pickle::*;
}

pub mod quantized {
    pub use candle_core::quantized::*;
}

pub mod safetensors {
    pub use candle_core::safetensors::*;
}

pub mod scalar {
    pub use candle_core::scalar::*;
}

pub mod shape {
    pub use candle_core::shape::*;
}

pub mod test_utils {
    pub use candle_core::test_utils::*;
}

pub mod utils {
    pub use candle_core::utils::*;
}

/// Curated re-exports of the most commonly used Candle functionality
///
/// This prelude includes the essential types and functions you'll need for most ML tasks,
/// without overwhelming you with the full API surface. For complete access, use the
/// individual modules or glob imports.
pub mod prelude {
    // Essential core types
    pub use candle_core::{
        // Common macros/functions
        bail,
        DType,
        Device,
        Error,
        IndexOp,
        Module,
        Result,
        Shape,
        Tensor,
        Var,
    };

    // Essential neural network components
    pub use candle_nn::{
        // Layer constructor functions
        embedding,
        layer_norm,
        linear,
        // Loss functions
        loss::{cross_entropy, mse},
        // Optimizers
        optim::{AdamW, Optimizer, ParamsAdamW, SGD},
        // Common layers
        Activation,
        Conv1d,
        Conv2d,
        Dropout,
        Embedding,
        // Initialization
        Init,
        LayerNorm,
        Linear,
        RmsNorm,
        // Variable management
        VarBuilder,
        VarMap,
    };

    // Backpropagation essentials
    pub use candle_core::backprop::GradStore;

    // Features enabled by default - always available unless opt-out
    #[cfg(feature = "flash-attn")]
    pub use candle_flash_attn::flash_attn;

    #[cfg(feature = "layer-norm")]
    pub use candle_layer_norm::{layer_norm as fused_layer_norm, rms_norm};
}
