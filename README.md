# Candlelight

**Unified wrapper for the Candle ML framework ecosystem**

Candlelight provides a single, coherent dependency for all Candle ML framework components, ensuring version compatibility and simplifying dependency management across projects.

## Why Candlelight?

The Candle ecosystem consists of multiple crates that must be kept in sync:
- `candle-core` - Core tensor operations
- `candle-nn` - Neural network layers  
- `candle-transformers` - Transformer implementations
- `candle-flash-attn` - FlashAttention-2 optimization
- `candle-layer-norm` - Fused kernel optimizations

Managing these across multiple projects is tedious and error-prone. Candlelight solves this by centralizing version management.

## Current Status: Git Snapshot

⚠️ **Temporary Git Dependency**: Currently using Candle git snapshot `db08cc0a` to support:
- CUDA 13.0 (via cudarc 0.17.1+)
- Visual Studio 2022 v17.12+ compatibility

Once Candle v0.10 is released with these features, we'll switch to stable crates.io releases.

## Features

### Hardware Acceleration
- `cuda` - NVIDIA GPU support via CUDA
- `metal` - Apple Silicon GPU support via Metal

### CUDA Optimizations
- `flash-attn` - FlashAttention-2 (requires `cuda`)
- `layer-norm` - Fused LayerNorm/RMSNorm kernels (requires `cuda`)
- `cuda-full` - All CUDA optimizations (convenience feature)

### Utilities
- `datasets` - Dataset loading utilities

## Installation

Add Candlelight to your `Cargo.toml`:

```toml
[dependencies]
# From GitHub (recommended until crates.io publication)
candlelight = { git = "https://github.com/ciresnave/candlelight", features = ["cuda-full"] }

# Or select specific features
candlelight = { git = "https://github.com/ciresnave/candlelight", features = ["cuda", "flash-attn"] }

# CPU-only
candlelight = { git = "https://github.com/ciresnave/candlelight" }
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
    let linear = Linear::new(vb.pp("fc"), 768, 256)?;
    let activation = Activation::Gelu;
    Ok((linear, activation))
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
candlelight = { git = "https://github.com/ciresnave/candlelight", features = ["cuda", "flash-attn"] }
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

- **CUDA Toolkit**: 12.4+ or 13.0+
- **Visual Studio**: 2022 (any version with latest Candle git snapshot)
- **GPU**: NVIDIA GPU with compute capability 6.0+

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
