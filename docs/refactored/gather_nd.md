# `torch::gather_nd` / `torch::gatherNd`

N-dimensional gather: return elements from `input` at the specified `indices` positions.

## Dual Syntax Support

### Positional (legacy)
```tcl
# torch::gather_nd input indices
set gathered [torch::gather_nd $inputTensor $indexTensor]
```

### Named (recommended)
```tcl
# torch::gather_nd -input handle -indices handle
set gathered [torch::gather_nd -input $inputTensor -indices $indexTensor]
```
The camelCase alias `torch::gatherNd` accepts identical parameters.

## Parameters
| Name | Positional Index | Description |
|------|------------------|-------------|
| `input` | 1 | Handle to the source tensor. |
| `indices` | 2 | Handle to an int64 tensor containing indices to gather. |

Both handles must reference existing tensors in the Tcl tensor storage (created earlier via any tensor-creation command).

## Examples
```tcl
# Create a source tensor
set src [torch::tensor_create {10 20 30 40} float32]
# Indices tensor (gather elements 3 and 1)
set idx [torch::tensor_create {3 1} int64]

# Positional syntax
set out1 [torch::gather_nd $src $idx]

# Named syntax
set out2 [torch::gather_nd -input $src -indices $idx]

# camelCase alias
set out3 [torch::gatherNd -input $src -indices $idx]
```

## Error Handling
The command returns descriptive errors when:
- Either handle is missing from storage (`Invalid input tensor`, `Invalid indices tensor`).
- Required parameters are omitted.
- Unknown named parameters are provided.

## Backward Compatibility
Existing positional syntax remains fully functional. The new named-parameter form and camelCase alias are additive.

## Tests
See `tests/refactored/gather_nd_test.tcl` for full coverage of:
1. Both syntaxes
2. camelCase alias
3. Error conditions

---
**Status:** Dual syntax implemented ✔️ | camelCase alias registered ✔️ | Tests ✔️ | Docs ✔️ 