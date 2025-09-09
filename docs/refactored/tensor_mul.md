# torch::tensor_mul / torch::tensorMul

Multiply two tensors element-wise.

---

## Dual Syntax Support

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_mul tensor1 tensor2
```

### Named Parameter Syntax (Modern)
```tcl
torch::tensor_mul -input tensor1 -other tensor2
torch::tensorMul -input tensor1 -other tensor2
```

---

## Parameters
| Name   | Type   | Required | Description                |
|--------|--------|----------|----------------------------|
| input  | str    | Yes      | Handle of first tensor     |
| other  | str    | Yes      | Handle of second tensor    |

---

## Return Value
- Returns a new tensor handle containing the element-wise product of the two input tensors.

---

## Examples

### Positional Syntax
```tcl
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set t2 [torch::tensor_create -data {2.0 3.0 4.0} -dtype float32]
set result [torch::tensor_mul $t1 $t2]
```

### Named Parameter Syntax
```tcl
set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
set t2 [torch::tensorCreate -data {2.0 3.0 4.0} -dtype float32]
set result [torch::tensor_mul -input $t1 -other $t2]
```

### camelCase Alias
```tcl
set result [torch::tensorMul -input $t1 -other $t2]
```

---

## Error Handling
- If either tensor handle is invalid, an error is raised: `Invalid first tensor name` or `Invalid second tensor name`.
- If required parameters are missing, an error is raised: `Required parameters missing: -input and -other`.
- If an unknown parameter is provided, an error is raised: `Unknown parameter: -param`.
- If a parameter value is missing, an error is raised: `Missing value for parameter`.

---

## Migration Guide
- **Old code (positional):**
  ```tcl
  torch::tensor_mul $t1 $t2
  ```
- **New code (named):**
  ```tcl
  torch::tensor_mul -input $t1 -other $t2
  # or
  torch::tensorMul -input $t1 -other $t2
  ```
- Both syntaxes are fully supported for backward compatibility.

---

## Test Coverage
- Both syntaxes
- camelCase alias
- Error handling (missing/invalid parameters)
- Edge cases (zero tensors, large values, different dtypes)
- Syntax consistency 