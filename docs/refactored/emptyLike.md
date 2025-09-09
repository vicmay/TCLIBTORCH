# torch::empty_like / torch::emptyLike

Create an uninitialized tensor with the same shape as an existing tensor. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## 📝 **Syntax**

### **Positional (backward compatible)**
```
torch::empty_like input ?dtype? ?device?
```

### **Named parameters (modern)**
```
torch::empty_like -input input ?-dtype dtype? ?-device device? ?-requiresGrad bool?
torch::emptyLike   -input input ?-dtype dtype? ?-device device? ?-requiresGrad bool?
```

---

## 🧩 **Parameters**
| Name         | Type   | Required | Default   | Description                                 |
|--------------|--------|----------|-----------|---------------------------------------------|
| input        | string | Yes      | -         | Handle of the input tensor                  |
| dtype        | string | No       | input's    | Data type (float32, float64, int32, int64)  |
| device       | string | No       | input's    | Device ("cpu" or "cuda")                    |
| requiresGrad | bool   | No       | false      | Whether tensor requires gradients           |

---

## 🏷️ **Aliases**
- `torch::empty_like` (snake_case)
- `torch::emptyLike` (camelCase)

---

## 📋 **Examples**

### **Positional Syntax**
```tcl
;# Basic usage
set t [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}}]
set result [torch::empty_like $t]

;# With dtype
set result [torch::empty_like $t float64]

;# With device
set result [torch::empty_like $t float32 cpu]
```

### **Named Parameter Syntax**
```tcl
set t [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}}]
set result [torch::empty_like -input $t -dtype float64 -device cpu -requiresGrad true]
```

### **CamelCase Alias**
```tcl
set t [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}}]
set result [torch::emptyLike -input $t -dtype float32]
```

---

## ⚠️ **Error Handling**
- Missing input: returns "Input tensor is required"
- Invalid tensor name: returns "Invalid tensor name"
- Too many positional arguments: returns "Invalid number of arguments"
- Named parameter without value: returns "Missing value for parameter"
- Unknown named parameter: returns "Unknown parameter: ..."
- Invalid requiresGrad value: returns "Invalid requiresGrad value: ..."

---

## 🔄 **Migration Guide**
- **Old (positional):**
  ```tcl
  torch::empty_like $t float32 cpu
  ```
- **New (named):**
  ```tcl
  torch::empty_like -input $t -dtype float32 -device cpu
  torch::emptyLike -input $t -dtype float32 -device cpu
  ```

Both syntaxes are fully supported. Migration is optional but recommended for clarity and future compatibility.

---

## ✅ **Test Coverage**
- Positional and named syntax
- CamelCase alias
- Error handling
- Different input shapes and data types
- Edge cases
- Syntax consistency

---

## 🔗 **See Also**
- [torch::zeros_like](zeros_like.md)
- [torch::ones_like](ones_like.md)
- [torch::full_like](full_like.md)
- [torch::empty](empty.md) 