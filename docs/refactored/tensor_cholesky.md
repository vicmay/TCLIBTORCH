# torch::tensor_cholesky / torch::tensorCholesky

Computes the Cholesky decomposition of a symmetric positive-definite matrix.

---

## 📝 **Syntax**

### **Positional (Backward Compatible)**
```tcl
torch::tensor_cholesky tensor
```

### **Named Parameters (Modern)**
```tcl
torch::tensor_cholesky -input tensor
# or
torch::tensor_cholesky -tensor tensor
```

---

## 📋 **Parameters**
| Name   | Type   | Required | Description                |
|--------|--------|----------|----------------------------|
| tensor | handle | Yes      | Input symmetric positive-definite matrix |

---

## 🔄 **Return Value**
A new tensor handle containing the lower triangular Cholesky factor L, where A = L * L^T.

---

## 🚦 **Examples**

### **Positional Syntax**
```tcl
# Create a 2x2 positive definite matrix
set t [torch::tensor_create {4.0 2.0 2.0 5.0} {2 2}]
set L [torch::tensor_cholesky $t]
# L contains the lower triangular Cholesky factor
```

### **Named Parameter Syntax**
```tcl
# Create a 3x3 positive definite matrix
set t [torch::tensor_create {4.0 12.0 -16.0 12.0 37.0 -43.0 -16.0 -43.0 98.0} {3 3}]
set L [torch::tensor_cholesky -input $t]
```

### **CamelCase Alias**
```tcl
# Using the camelCase alias
set L [torch::tensorCholesky -input $t]
```

---

## 🛠️ **Migration Guide**
| Old (positional)                | New (named)                  |
|---------------------------------|------------------------------|
| torch::tensor_cholesky $t       | torch::tensor_cholesky -input $t |
| torch::tensor_cholesky $t       | torch::tensorCholesky -input $t |

---

## ⚠️ **Error Handling**
- If the tensor handle is missing or invalid, returns `Invalid tensor name`.
- If required parameters are missing, returns `Usage: torch::tensor_cholesky tensor | torch::tensor_cholesky -input tensor`.
- If an unknown parameter is provided, returns `Unknown parameter: -param. Valid parameters are: -input, -tensor`.
- If a parameter value is missing, returns `Missing value for parameter`.
- If the input matrix is not positive definite, PyTorch will throw an appropriate error.

---

## 🧪 **Testing**
- Both syntaxes are tested for correctness and error handling.
- Various matrix sizes are tested (2x2, 3x3).
- Error conditions and edge cases are validated.

---

## 📝 **Notes**
- The input matrix must be symmetric and positive definite.
- The result is a lower triangular matrix L such that A = L * L^T.
- The Cholesky decomposition is unique for positive definite matrices.
- 100% backward compatibility is maintained.
- Both `torch::tensor_cholesky` and `torch::tensorCholesky` are available.
- This operation is commonly used in linear algebra, statistics, and optimization. 