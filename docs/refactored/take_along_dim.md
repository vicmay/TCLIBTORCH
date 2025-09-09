# torch::take_along_dim / torch::takeAlongDim

Take values from a tensor along a specified dimension using indices. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## 📝 **Syntax**

### **Positional (backward compatible)**
```
torch::take_along_dim input indices ?dim?
```

### **Named parameters (modern)**
```
torch::take_along_dim -input input -indices indices ?-dim dim?
torch::takeAlongDim   -input input -indices indices ?-dim dim?
```

---

## 🧩 **Parameters**
| Name     | Type    | Required | Description                                 |
|----------|---------|----------|---------------------------------------------|
| input    | tensor  | Yes      | Input tensor handle                         |
| indices  | tensor  | Yes      | Indices tensor handle (int64)               |
| dim      | int     | No       | Dimension to take along (default: inferred) |

---

## 🏷️ **Aliases**
- `torch::take_along_dim` (snake_case)
- `torch::takeAlongDim` (camelCase)

---

## 📋 **Examples**

### **Positional Syntax**
```tcl
set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32]
set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64]
set result [torch::take_along_dim $input $indices]
```

### **Named Parameter Syntax**
```tcl
set result [torch::take_along_dim -input $input -indices $indices -dim 1]
```

### **CamelCase Alias**
```tcl
set result [torch::takeAlongDim -input $input -indices $indices -dim 1]
```

---

## ⚠️ **Error Handling**
- Missing required parameters: returns usage message
- Invalid tensor handles: returns "Invalid input tensor" or "Invalid indices tensor"
- Invalid dimension: returns "Invalid dim value"
- Unknown named parameter: returns "Unknown parameter: ..."

---

## 🔄 **Migration Guide**
- **Old (positional):**
  ```tcl
  torch::take_along_dim input indices ?dim?
  ```
- **New (named):**
  ```tcl
  torch::take_along_dim -input input -indices indices ?-dim dim?
  torch::takeAlongDim   -input input -indices indices ?-dim dim?
  ```

Both syntaxes are fully supported. Migration is optional but recommended for clarity and future compatibility.

---

## ✅ **Test Coverage**
- Positional and named syntax
- CamelCase alias
- Error handling
- Mathematical correctness
- Edge cases
- Syntax consistency

---

## 🔗 **See Also**
- [torch::gather_nd](gather_nd.md)
- [torch::scatter_nd](scatter_nd.md) 