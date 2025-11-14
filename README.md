# Candlelight

**Unified wrapper for the Candle ML framework ecosystem**

Candlelight provides a single, coherent dependency for all Candle ML framework components, ensuring version compatibility and simplifying dependency management across projects.

## Why Candlelight?

The Candle ecosystem consists of multiple crates that must be kept in sync:
- `candle-core` - Core tensor operations
- `candle-nn` - Neural network layers  
- `candle-transformers` - Transformer implementations
- `candle-flash-attn` - FlashAttention-2 optimization
- `candle-layer-norm` - Fused LayerNorm/RMSNorm kernels
- `candle-datasets` - Dataset loading utilities
- `candle-optimisers` - Advanced optimizers (AdamW, Lion, etc.)
- `candle-bhop` - Basin-hopping global optimization

Managing these across multiple projects is tedious and error-prone. Candlelight solves this by centralizing version management.

## Current Status: Git Snapshot

⚠️ **Temporary Git Dependencies**: Currently using git snapshots to support:
- **Candle** (git rev `db08cc0a`) - CUDA 13.0 support via cudarc 0.17.8+
- **candle-layer-norm** - [Fork](https://github.com/ciresnave/candle-layer-norm) with CUDA 13.0 + cudarc 0.17.8 + Windows MSVC fixes ([PR #2](https://github.com/EricLBuehler/candle-layer-norm/pull/2))
- **candle-optimisers** - [Fork](https://github.com/ciresnave/candle-optimisers) updated for Candle v0.9.2-alpha.1 ([PR #29](https://github.com/KGrewal1/optimisers/pull/29))
- **candle-bhop** - [Fork](https://github.com/ciresnave/candle-bhop) updated for Candle v0.9.2-alpha.1 ([PR #1](https://github.com/KGrewal1/candle-bhop/pull/1))

Once upstream PRs are merged and Candle v0.10 is released, we'll switch to stable crates.io releases.

## Philosophy: "It Just Works"

All features are **enabled by default** for the best out-of-box experience. Users can opt-out with `default-features = false` if they need a minimal configuration.

## Features

### Default Features (Enabled Automatically)
Following the "it just works" philosophy, these features are enabled by default:
- ✅ `flash-attn` - FlashAttention-2 optimization (requires CUDA)
- ✅ `layer-norm` - Fused LayerNorm/RMSNorm kernels (requires CUDA)
- ✅ `cudnn` - cuDNN optimizations (requires cuDNN installation)
- ✅ `datasets` - Dataset loading utilities
- ✅ `optimizers` - Advanced optimizers (AdamW, Lion, Sophia, etc.)
- ✅ `basin-hopping` - Basin-hopping global optimization

### Hardware Acceleration
- `cuda` - NVIDIA GPU support via CUDA (auto-enabled by default features)
- `metal` - Apple Silicon GPU support via Metal

### Additional Features
- `cuda-full` - All CUDA optimizations (convenience feature, redundant with defaults)

## Installation

Add Candlelight to your `Cargo.toml`:

```toml
[dependencies]
# From GitHub (recommended until crates.io publication)
# All features enabled by default for "it just works" experience
candlelight = { git = "https://github.com/ciresnave/candlelight" }

# Minimal CPU-only configuration
candlelight = { git = "https://github.com/ciresnave/candlelight", default-features = false }

# CPU with specific features
candlelight = { git = "https://github.com/ciresnave/candlelight", default-features = false, features = ["datasets"] }
```

## Usage

### In your code

```rust
use candlelight::{Device, Tensor, Result};

fn main() -> Result<()> {
    // Use CUDA if available, fall back to CPU
    let device = Device::cuda_if_available(0)?;
    
    // Create a random tensor
    let x = Tensor::randn(0f32, 1.0, (128, 768), &device)?;
    
    println!("Tensor shape: {:?}", x.shape());
    Ok(())
}
```

### Using the prelude

```rust
use candlelight::prelude::*;

fn build_model(vb: VarBuilder) -> Result<impl Module> {
    let linear = linear(768, 256, vb.pp("fc"))?;
    Ok(linear)
}
```

### Using optimizers

```rust
use candlelight::prelude::*;
use candlelight::optimizers::adamw;

fn train(model: &impl Module, data: &Tensor) -> Result<()> {
    let mut optimizer = adamw(model.parameters(), Default::default())?;
    
    // Training loop
    for epoch in 0..10 {
        let loss = model.forward(data)?;
        optimizer.backward_step(&loss)?;
    }
    Ok(())
}
```

## Migration from Direct Candle Dependencies

### Before:
```toml
candle-core = { git = "...", rev = "..." }
candle-nn = { git = "...", rev = "..." }
candle-transformers = { git = "...", rev = "..." }
candle-flash-attn = { git = "...", rev = "...", optional = true }
```

### After:
```toml
# All optimizations enabled by default!
candlelight = { git = "https://github.com/ciresnave/candlelight" }
```

### Code changes:
```rust
// Before:
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

// After:
use candlelight::{Device, Tensor};
use candlelight::nn::VarBuilder;
// Or use the prelude:
use candlelight::prelude::*;
```

## CUDA Requirements

- **CUDA Toolkit**: 12.4+ or 13.0+ (13.0+ recommended)
- **cuDNN**: 8.9+ or 9.x (optional but recommended for `cudnn` feature)
- **Visual Studio**: 2022 v17.12+ with MSVC toolchain (Windows)
- **GPU**: NVIDIA GPU with compute capability 6.0+

### Windows MSVC Notes
Candlelight includes fixes for Windows MSVC compatibility:
- Large object compilation support (`/bigobj`)
- Proper C++ standard library linking (no `stdc++.lib` errors)

## Repository

**GitHub**: <https://github.com/ciresnave/candlelight>

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

This is a wrapper crate. For issues with underlying Candle functionality, please report to the [Candle repository](https://github.com/huggingface/candle).

For Candlelight-specific issues (feature configuration, re-exports, documentation), please open an issue at <https://github.com/ciresnave/candlelight/issues>.
