# torch::empty_cache / torch::emptyCache

Empty the CUDA memory cache to free up GPU memory. Supports both positional and named parameter syntax, as well as a camelCase alias.

---

## 📝 **Syntax**

### **Positional (backward compatible)**
```
torch::empty_cache ?device?
```

### **Named parameters (modern)**
```
torch::empty_cache ?-device device?
torch::emptyCache   ?-device device?
```

---

## 🧩 **Parameters**
| Name   | Type   | Required | Default | Description                    |
|--------|--------|----------|---------|--------------------------------|
| device | string | No       | ""      | Device to clear cache for      |

---

## 🏷️ **Aliases**
- `torch::empty_cache` (snake_case)
- `torch::emptyCache` (camelCase)

---

## 📋 **Examples**

### **Positional Syntax**
```tcl
;# Clear cache for all devices
set result [torch::empty_cache]

;# Clear cache for specific device
set result [torch::empty_cache cpu]
set result [torch::empty_cache cuda:0]
```

### **Named Parameter Syntax**
```tcl
;# Clear cache for all devices
set result [torch::empty_cache]

;# Clear cache for specific device
set result [torch::empty_cache -device cpu]
set result [torch::empty_cache -device cuda:0]
```

### **CamelCase Alias**
```tcl
;# Using camelCase alias
set result [torch::emptyCache]
set result [torch::emptyCache -device cuda:0]
```

---

## ⚠️ **Error Handling**
- Too many positional arguments: returns "Invalid number of arguments"
- Named parameter without value: returns "Missing value for parameter"
- Unknown named parameter: returns "Unknown parameter: ..."

---

## 🔄 **Migration Guide**
- **Old (positional):**
  ```tcl
  torch::empty_cache cpu
  ```
- **New (named):**
  ```tcl
  torch::empty_cache -device cpu
  torch::emptyCache -device cpu
  ```

Both syntaxes are fully supported. Migration is optional but recommended for clarity and future compatibility.

---

## 📊 **Return Values**
| Value                | Description                                    |
|----------------------|------------------------------------------------|
| `cache_cleared`      | Cache was successfully cleared                 |
| `cuda_not_available` | CUDA is not available on this system          |
| `cache_clear_attempted` | Cache clear was attempted but may have failed |

---

## ✅ **Test Coverage**
- Positional and named syntax
- CamelCase alias
- Error handling
- Different devices (CPU, CUDA)
- Edge cases
- Syntax consistency
- CUDA availability handling

---

## 🔗 **See Also**
- [torch::synchronize](synchronize.md)
- [torch::memory_stats](memory_stats.md)
- [torch::memory_summary](memory_summary.md) 