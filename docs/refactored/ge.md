# `torch::ge` / `torch::ge`

Element-wise greater-or-equal comparison between two tensors.

## Usage

### Positional (legacy)
```tcl
# torch::ge tensor1 tensor2
set result [torch::ge $a $b]
```

### Named (recommended)
```tcl
# torch::ge -input1 tensor1 -input2 tensor2
set result [torch::ge -input1 $a -input2 $b]
```

Returns a Boolean tensor (uint8) with 1 where `tensor1 >= tensor2` and 0 otherwise.  
The output tensor keeps the broadcasted shape of the inputs.

## Parameters
| Name | Positional Index | Description |
|------|------------------|-------------|
| `input1` | 1 | First tensor handle. |
| `input2` | 2 | Second tensor handle. |

## Examples
```tcl
set a [torch::tensor_create {1 2 3} float32]
set b [torch::tensor_create {2} float32]
set out [torch::ge -input1 $a -input2 $b]
```

## Error Handling
Raises descriptive errors if:
- Any tensor handle is invalid.
- Required parameters are missing.
- Unknown named parameters are supplied.

## Backward Compatibility
Positional syntax remains unchanged; named-parameter syntax is additive.

## Tests
See `tests/refactored/ge_test.tcl` for full coverage. 