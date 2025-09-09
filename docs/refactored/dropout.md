# `torch::dropout`

Creates a dropout layer module for regularization during training.
Supports both legacy positional syntax and the new named-parameter API.

---
## 📜 Legacy Positional Syntax (Backward-Compatible)
```tcl
# torch::dropout ?p? ?training? ?inplace?
set dropout_layer [torch::dropout 0.5 true false]
```

## 🆕 Named-Parameter Syntax
```tcl
set dropout_layer [torch::dropout \
    -p         0.5     ;# dropout probability (default 0.5)
    -training  true    ;# training mode (default true)
    -inplace   false   ;# inplace operation (default false)
]
```

---
## Parameters
| Name | Positional Index | Named Flag | Type | Default | Description |
|------|------------------|------------|------|---------|-------------|
| p | 1 | `-p` | double | `0.5` | Probability of an element to be zeroed |
| training | 2 | `-training` | bool | `true` | Whether layer is in training mode |
| inplace | 3 | `-inplace` | bool | `false` | If true, do operation in-place |

### Parameter Details
- **p**: Must be between 0.0 and 1.0 (inclusive)
- **training**: Accepts `true`/`false` or `1`/`0`  
- **inplace**: Accepts `true`/`false` or `1`/`0`

---
## Return Value
A module handle (string) that can be used with other layer operations like `torch::layer_forward`.

---
## Examples

### Basic Usage (Positional)
```tcl
# Create dropout with default 50% probability
set dropout [torch::dropout]

# Create dropout with 30% probability
set dropout [torch::dropout 0.3]

# Full positional specification
set dropout [torch::dropout 0.2 true false]
```

### Basic Usage (Named)
```tcl
# Create dropout with named parameters
set dropout [torch::dropout -p 0.3]

# All parameters specified
set dropout [torch::dropout -p 0.2 -training true -inplace false]

# Mixed parameter order (named syntax allows this)
set dropout [torch::dropout -training false -p 0.1 -inplace true]
```

### Using with Forward Pass
```tcl
# Create input tensor and dropout layer
set x [torch::randn -shape {32 128}]
set dropout [torch::dropout -p 0.5]

# Apply dropout during forward pass
set y [torch::layer_forward $dropout $x]
```

---
## Migration Guide
1. **Simple case**: `torch::dropout 0.3` → `torch::dropout -p 0.3`
2. **Multiple params**: `torch::dropout 0.2 true false` → `torch::dropout -p 0.2 -training true -inplace false`
3. **Advantage**: Named syntax allows any parameter order and partial specification

---
## Error Handling
The command validates:
* `p` is between 0.0 and 1.0
* Parameter values are correctly typed
* Named parameters come in pairs
* No unknown parameter names

Common errors:
```tcl
# ❌ Invalid probability
torch::dropout -p 1.5  ;# Error: p must be between 0.0 and 1.0

# ❌ Missing parameter value
torch::dropout -p       ;# Error: Named parameters must come in pairs

# ❌ Unknown parameter
torch::dropout -rate 0.3  ;# Error: Unknown parameter: -rate
```

---
## Implementation Notes
- Creates a PyTorch `nn.Dropout` module internally
- The `training` parameter controls dropout behavior (active in training mode only)
- The `inplace` parameter determines whether the operation modifies the input tensor
- Module handles are managed automatically and can be reused

---
## Test Coverage
See `tests/refactored/dropout_test.tcl` for:
* Positional & named syntax functionality
* Parameter validation & error cases
* Boolean parameter variations
* Edge cases and boundary values
* Module creation verification

---
© LibTorch TCL Extension – Dual Syntax API Modernization 