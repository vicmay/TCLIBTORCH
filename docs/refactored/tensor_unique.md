# torch::tensor_unique / torch::tensorUnique

Returns the unique elements of a tensor. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## **Syntax**

### **Positional (backward compatible)**
```tcl
# Only tensor name (sorted=true, return_inverse=false by default)
torch::tensor_unique tensor

# Specify sorted (0/1)
torch::tensor_unique tensor sorted

# Specify sorted and return_inverse
torch::tensor_unique tensor sorted return_inverse
```

### **Named Parameters (modern)**
```tcl
torch::tensor_unique -tensor tensorName ?-sorted bool? ?-returnInverse bool?
```

### **CamelCase Alias**
```tcl
torch::tensorUnique ...
```

---

## **Parameters**
| Name            | Type    | Required | Default | Description                                 |
|-----------------|---------|----------|---------|---------------------------------------------|
| tensor / -tensor| string  | yes      |         | Name of the input tensor                    |
| sorted / -sorted| bool    | no       | true    | Whether to return sorted unique values      |
| return_inverse / -returnInverse | bool | no | false   | Whether to return the inverse indices      |

---

## **Return Value**
- If `return_inverse` is false: returns the name of the tensor containing unique values.
- If `return_inverse` is true: returns a Tcl list `{unique uniqueTensorName inverse inverseTensorName}`.

---

## **Examples**

### **Positional Syntax**
```tcl
# Unique values (sorted)
torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0 4.0} -dtype float32
set result [torch::tensor_unique tensor0]
# result => tensor1 (unique values: 1.0 2.0 3.0 4.0)
```

```tcl
# Unique values, preserve order (sorted=0)
torch::tensor_create -data {3.0 1.0 2.0 1.0 4.0 2.0} -dtype float32
set result [torch::tensor_unique tensor0 0]
# result => tensor1 (unique values: 3.0 1.0 2.0 4.0)
```

```tcl
# Unique values and inverse indices
torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0} -dtype float32
set result [torch::tensor_unique tensor0 1 1]
# result => {unique tensor1 inverse tensor2}
```

### **Named Parameter Syntax**
```tcl
# Unique values
torch::tensor_unique -tensor tensor0
# Unique values, preserve order
torch::tensor_unique -tensor tensor0 -sorted 0
# Unique values and inverse indices
torch::tensor_unique -tensor tensor0 -sorted 1 -returnInverse 1
```

### **CamelCase Alias**
```tcl
set result [torch::tensorUnique -tensor tensor0 -returnInverse 1]
```

---

## **Error Handling**
- Missing required parameters: returns an error message.
- Invalid tensor name: returns "Tensor not found".
- Invalid parameter: returns "Unknown parameter: ...".
- Missing parameter value: returns "Missing value for parameter".

---

## **Migration Guide**
- **Old:** `torch::tensor_unique tensor0 0 1`
- **New:** `torch::tensor_unique -tensor tensor0 -sorted 0 -returnInverse 1`
- **CamelCase:** `torch::tensorUnique -tensor tensor0 -sorted 0 -returnInverse 1`

---

## **See Also**
- [torch::tensor_to_list](tensor_to_list.md)
- [torch::tensor_create](tensor_create.md)

---

## **Test Coverage**
- Both syntaxes, camelCase alias, error handling, edge cases, and parameter validation are covered in `tests/refactored/tensor_unique_test.tcl`. 