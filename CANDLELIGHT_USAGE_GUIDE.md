# Candlelight Usage Guide: Complete Candle Ecosystem Access

Candlelight provides comprehensive re-exports of the entire Candle ML framework ecosystem through a single, unified interface. **All public APIs from all Candle crates are available** through Candlelight with zero runtime overhead.

## Quick Reference

### Import Patterns

```rust
// 🎯 Most Common: Get everything you need
use candlelight::prelude::*;

// 🚀 Maximum Access: Everything from core + essentials
use candlelight::*;

// 🎛️ Module-Specific: Target specific functionality
use candlelight::nn::*;
use candlelight::transformers_models::*;
use candlelight::backprop::*;

// 🔧 Selective: Cherry-pick what you need
use candlelight::{Tensor, Device, DType, Result};
use candlelight::nn::{Linear, VarBuilder, AdamW};
```

## Complete Module Map

| Candlelight Module                    | Contains                              | Original Crate          |
| ------------------------------------- | ------------------------------------- | ----------------------- |
| `candlelight::*`                      | **Everything** from core + essentials | `candle_core`           |
| `candlelight::nn::*`                  | Complete neural network functionality | `candle_nn`             |
| `candlelight::transformers_models::*` | All transformer models & utilities    | `candle_transformers`   |
| `candlelight::backprop::*`            | Gradient computation & storage        | `candle_core::backprop` |
| `candlelight::flash_attention::*`     | FlashAttention-2 optimizations        | `candle_flash_attn` ¹   |
| `candlelight::fused_ops::*`           | Fused kernels (LayerNorm, RMS)        | `candle_layer_norm` ¹   |
| `candlelight::data::*`                | Dataset loading utilities             | `candle_datasets` ¹     |
| `candlelight::prelude::*`             | Curated essentials for most tasks     | Multiple                |

¹ *Available when corresponding feature is enabled*

## Real-World Usage Examples

### 🏗️ Building a Neural Network

```rust
use candlelight::prelude::*;

fn create_model(device: &Device) -> Result<impl Module> {
    let vs = VarBuilder::from_tensors(VarMap::new(), DType::F32, device);
    
    Ok(Linear::new(
        Tensor::randn(0.0, 1.0, (784, 128), device)?,
        Some(Tensor::zeros((128,), device)?),
    ))
}

// Alternative: Access everything directly
use candlelight::*;
// Now you have: Tensor, Device, Linear, VarBuilder, etc.
```

### 🤖 Working with Transformers

```rust
use candlelight::transformers_models::*;
use candlelight::{Device, DType};

fn load_bert_model() -> Result<()> {
    // All transformer models available directly:
    // BertModel, GPT2Model, LlamaModel, etc.
    let config = BertConfig::default();
    // ... model loading code
    Ok(())
}
```

### ⚡ Optimized Operations (with features)

```rust
// Cargo.toml: features = ["flash-attn", "layer-norm"]
use candlelight::flash_attention::*;
use candlelight::fused_ops::*;

fn optimized_attention() -> Result<()> {
    // FlashAttention-2 available directly
    let output = flash_attn(query, key, value, None, false, false)?;
    
    // Fused LayerNorm available directly  
    let normalized = layer_norm(input, &weight, &bias, 1e-5)?;
    
    Ok(())
}
```

### 🎯 Training Loop with Optimizers

```rust
use candlelight::prelude::*;

fn training_example() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let vs = VarBuilder::from_tensors(VarMap::new(), DType::F32, &device);
    
    // All optimizers available directly
    let mut optimizer = AdamW::new(
        vs.all_vars(),
        ParamsAdamW { lr: 1e-3, ..Default::default() }
    )?;
    
    // GradStore for gradient management
    let mut grad_store = GradStore::new();
    
    // Training loop...
    Ok(())
}
```

## 🔧 Advanced Access Patterns

### Direct Crate Access (Alternative)
```rust
// Access original crates directly if preferred
use candlelight::core;        // candle_core
use candlelight::nn;          // candle_nn  
use candlelight::transformers; // candle_transformers

let tensor = core::Tensor::randn(0.0, 1.0, (2, 3), &device)?;
let linear = nn::Linear::new(weight, bias);
```

### Mixing Import Styles
```rust
// Combine approaches as needed
use candlelight::prelude::*;  // Get essentials
use candlelight::transformers_models::BertModel;  // Add specific models
use candlelight::flash_attention::flash_attn;     // Add optimizations
```

## 🎛️ Feature-Dependent Functionality

Enable features in your `Cargo.toml`:

```toml
[dependencies]
candlelight = { 
    path = "../candlelight", 
    features = ["cuda", "flash-attn", "layer-norm", "datasets"] 
}
```

Then access feature-specific modules:
- `candlelight::flash_attention::*` (requires `flash-attn`)  
- `candlelight::fused_ops::*` (requires `layer-norm`)
- `candlelight::data::*` (requires `datasets`)

## 💡 Key Benefits for Your Projects

1. **🎯 Single Dependency**: Replace all individual Candle crates with just `candlelight`
2. **🚀 Zero Overhead**: Unused exports are eliminated by Rust's compiler
3. **🔄 Version Sync**: All Candle components guaranteed compatible
4. **📚 Complete Access**: Every public API from the Candle ecosystem available
5. **🎨 Flexible**: Choose your preferred import style

## 🚨 Migration from Individual Candle Crates

Replace this:
```toml
# OLD: Multiple dependencies
candle-core = "0.x"
candle-nn = "0.x" 
candle-transformers = "0.x"
```

With this:
```toml  
# NEW: Single dependency
candlelight = { path = "../candlelight", features = ["cuda"] }
```

Update imports:
```rust
// OLD
use candle_core::{Tensor, Device};
use candle_nn::{Linear, VarBuilder};

// NEW: Same functionality, cleaner imports
use candlelight::prelude::*;
// or
use candlelight::{Tensor, Device};
use candlelight::nn::{Linear, VarBuilder};
```

## 🎓 Recommendation for LLM-Assisted Development

When working with LLMs on Candlelight projects, inform them:

> "This project uses Candlelight, which provides complete access to the entire Candle ML framework ecosystem. All types, functions, and modules from candle-core, candle-nn, candle-transformers, and optional crates are available through candlelight imports. Use `candlelight::prelude::*` for most tasks, or access specific modules like `candlelight::nn::*` and `candlelight::transformers_models::*` as needed."

This ensures LLMs understand they have full access to Candle's functionality through the unified Candlelight interface.