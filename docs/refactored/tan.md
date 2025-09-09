# torch::tan / torch::tan

Computes the element-wise tangent of the input tensor.

---

## 📝 **Syntax**

### **Positional (Backward Compatible)**
```tcl
torch::tan tensor
```

### **Named Parameters (Modern)**
```tcl
torch::tan -input tensor
# or
torch::tan -tensor tensor
```

---

## 📋 **Parameters**
| Name      | Type   | Required | Description                |
|-----------|--------|----------|----------------------------|
| tensor    | handle | Yes      | Input tensor handle        |

---

## 🔄 **Return Value**
A new tensor handle containing the element-wise tangent of the input.

---

## 🚦 **Examples**

### **Positional Syntax**
```tcl
set t [torch::tensor_create {0.0 0.7853981633974483 1.5707963267948966} float32]
set result [torch::tan $t]
# result is a tensor with values {0.0 1.0 large}
```

### **Named Parameter Syntax**
```tcl
set t [torch::tensor_create {0.0 0.7853981633974483 1.5707963267948966} float32]
set result [torch::tan -input $t]
```

---

## 🛠️ **Migration Guide**
| Old (positional)                | New (named)                  |
|---------------------------------|------------------------------|
| torch::tan $t                   | torch::tan -input $t         |
| torch::tan $t                   | torch::tan -tensor $t        |

---

## ⚠️ **Error Handling**
- If the tensor handle is missing or invalid, returns `Invalid tensor name`.
- If required parameters are missing, returns `Usage: torch::tan tensor | torch::tan -input tensor`.
- If an unknown parameter is provided, returns `Unknown parameter: -param. Valid parameters are: -input, -tensor`.
- If a parameter value is missing, returns `Missing value for parameter`.

---

## 🧪 **Testing**
- Both syntaxes are tested for correctness and error handling.
- Edge cases (e.g., tan(pi/2)) are tested for large output values.

---

## 📝 **Notes**
- The tangent of pi/2 is mathematically undefined (infinite); the result will be a very large value due to floating-point limitations.
- Both `-input` and `-tensor` are accepted as named parameters for compatibility.
- 100% backward compatibility is maintained. 