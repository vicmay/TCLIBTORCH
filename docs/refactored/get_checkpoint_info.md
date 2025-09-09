# `torch::get_checkpoint_info` / `torch::getCheckpointInfo`

Retrieve metadata (epoch, loss, learning rate, etc.) stored inside a checkpoint file created with `torch::save_checkpoint`.

---

## Named-parameter Syntax (Preferred)

```tcl
torch::getCheckpointInfo -file checkpoint.pt
```

Parameters:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `-file` / `-filename` | string | ✅ | Path to the checkpoint file |

---

## Positional Syntax (Backward-compatible)

```tcl
torch::get_checkpoint_info checkpoint.pt
```

---

## Examples

```tcl
# Create a checkpoint (assumes model/optimizer previously defined)
torch::save_checkpoint myModel myOpt 5 0.1234 0.001 checkpoint.pt

# Query information
set info [torch::getCheckpointInfo -file checkpoint.pt]
puts $info ;# => {epoch 5 loss 0.1234 learning_rate 0.001}
```

---

## Error Handling
| Condition | Message |
|-----------|---------|
| File not found / invalid path | `Checkpoint file not found` |
| Missing parameter value | `Missing value for option -file` |
| Unknown parameter | `Unknown parameter: -foo` |

---

## Tests
See `tests/refactored/get_checkpoint_info_test.tcl` for exhaustive test cases covering both syntaxes and error scenarios. 