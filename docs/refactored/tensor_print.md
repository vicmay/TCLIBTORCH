# `torch::tensor_print` / `torch::tensorPrint`

Prints a string representation of a tensor. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## 📝 **Usage**

### **Positional Syntax (Backward Compatible)**
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
set result [torch::tensor_print $reshaped]
puts $result
```

### **Named Parameter Syntax (New)**
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
set result [torch::tensor_print -input $reshaped]
puts $result
```

### **CamelCase Alias**
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
set result [torch::tensorPrint $reshaped]
puts $result
```

---

## 🧾 **Parameters**

| Name   | Type      | Required | Description                |
|--------|-----------|----------|----------------------------|
| input  | string    | Yes      | Tensor handle to print     |

- **Positional:** `torch::tensor_print <tensor>`
- **Named:** `torch::tensor_print -input <tensor>`

---

## ✅ **Return Value**
A string representation of the tensor, including its shape, dtype, and values.

---

## ⚠️ **Error Handling**
- **Invalid tensor name:**
  - Returns: `Invalid tensor name`
- **Missing required parameter:**
  - Returns: `Input tensor is required`
- **Unknown parameter:**
  - Returns: `Unknown parameter: -param`
- **Too many arguments (positional):**
  - Returns: `Invalid number of arguments`
- **Missing value for parameter:**
  - Returns: `Missing value for parameter`

---

## 🔄 **Migration Guide**
- **Old (positional):**
  - `torch::tensor_print $tensor`
- **New (named):**
  - `torch::tensor_print -input $tensor`
- **CamelCase:**
  - `torch::tensorPrint $tensor`

All syntaxes are fully supported and produce identical results.

---

## 🧪 **Examples**

### Print a 2x2 float tensor
```tcl
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
set t2 [torch::tensor_reshape -input $t -shape {2 2}]
puts [torch::tensor_print $t2]
```

### Print using named parameter
```tcl
puts [torch::tensor_print -input $t2]
```

### Print using camelCase alias
```tcl
puts [torch::tensorPrint $t2]
```

---

## 🧩 **Edge Cases**
- Works for empty, single-element, and large tensors
- Supports all dtypes and devices

---

## 🛡️ **Test Coverage**
- Both syntaxes, camelCase alias
- Error handling (invalid tensor, missing/extra/unknown parameters)
- Edge cases (empty, large, int/double tensors)
- Syntax consistency
- Mathematical correctness

---

## 📚 **See Also**
- [`tensor_create`](tensor_create.md)
- [`tensor_reshape`](tensor_reshape.md)
- [`tensor_numel`](tensor_numel.md) 