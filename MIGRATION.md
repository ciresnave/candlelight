# Migration Guide: Candle → Candlelight

This guide helps you migrate from direct Candle dependencies to Candlelight.

## Step 1: Update Cargo.toml

### Remove old dependencies:
```toml
# Remove these:
candle-core = { git = "...", ... }
candle-nn = { git = "...", ... }
candle-transformers = { git = "...", ... }
candle-flash-attn = { git = "...", ... }
candle-layer-norm = { git = "...", ... }
```

### Add Candlelight:
```toml
candlelight = { workspace = true }
# Or if not using workspace:
candlelight = { path = "../crates/candlelight", features = ["cuda-full"] }
```

### Update features:
```toml
[features]
# Before:
cuda = ["candle-core/cuda", "candle-nn/cuda", ...]

# After:
cuda = ["candlelight/cuda"]
flash-attn = ["candlelight/flash-attn"]
cuda-full = ["candlelight/cuda-full"]
```

## Step 2: Update Code Imports

### Find and replace in all .rs files:

```bash
# Using PowerShell:
Get-ChildItem -Recurse -Filter *.rs | ForEach-Object {
    (Get-Content $_.FullName) -replace 'use candle_core', 'use candlelight::core' |
    Set-Content $_.FullName
}

Get-ChildItem -Recurse -Filter *.rs | ForEach-Object {
    (Get-Content $_.FullName) -replace 'use candle_nn', 'use candlelight::nn' |
    Set-Content $_.FullName
}

Get-ChildItem -Recurse -Filter *.rs | ForEach-Object {
    (Get-Content $_.FullName) -replace 'use candle_transformers', 'use candlelight::transformers' |
    Set-Content $_.FullName
}
```

### Or use the prelude (recommended for new code):
```rust
// Instead of:
use candle_core::{Device, Tensor, Result};
use candle_nn::VarBuilder;

// Use:
use candlelight::prelude::*;
```

## Step 3: Module Path Updates

Since Candlelight re-exports modules, you may need to update paths:

### Before:
```rust
use candle_transformers::models::llama::{Llama, Config};
```

### After:
```rust
use candlelight::transformers::models::llama::{Llama, Config};
```

## Common Patterns

### Pattern 1: Basic types
```rust
// Old:
use candle_core::{Device, Tensor, DType, Result};

// New (option 1 - explicit):
use candlelight::{Device, Tensor, DType, Result};

// New (option 2 - via module):
use candlelight::core::{Device, Tensor, DType, Result};

// New (option 3 - prelude):
use candlelight::prelude::*;
```

### Pattern 2: Neural network layers
```rust
// Old:
use candle_nn::{Linear, VarBuilder, Embedding};

// New:
use candlelight::nn::{Linear, VarBuilder, Embedding};
// Or:
use candlelight::prelude::*;
```

### Pattern 3: Transformers
```rust
// Old:
use candle_transformers::models::llama::Llama;

// New:
use candlelight::transformers::models::llama::Llama;
```

### Pattern 4: Flash attention (when feature enabled)
```rust
// Old:
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;

// New:
#[cfg(feature = "flash-attn")]
use candlelight::flash_attn::flash_attn;
// Or:
use candlelight::prelude::*; // Includes flash_attn when feature enabled
```

## Verification

After migration, verify everything compiles:

```bash
# Check syntax
cargo check

# Run tests
cargo test

# Build with CUDA
cargo build --features cuda-full
```

## Troubleshooting

### Issue: "Module not found"
- Check that you've updated the module path (e.g., `candle_core` → `candlelight::core`)
- Verify features are enabled in Cargo.toml if using optional dependencies

### Issue: "Type mismatch"
- Candlelight re-exports the same types, so this shouldn't happen
- If it does, you may have mixed old and new imports

### Issue: Build fails with git dependency error
- Ensure Candlelight is in your workspace or path is correct
- Check that Candlelight's Cargo.toml has the correct git rev
