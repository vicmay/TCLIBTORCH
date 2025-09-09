# LibTorch TCL Extension - One-by-One Refactoring Workflow

**Status**: 🚀 **READY TO START** - Systematic Command-by-Command Refactoring  
**Approach**: Refactor → Test → Document → Repeat  
**Quality**: Each command fully validated before moving to next  

---

## 🎯 **REFACTORING WORKFLOW - STEP BY STEP**

### **For Each Command, Follow This Complete Process:**

#### **STEP 1: PREPARATION**
1. **Select Command**: Choose next command from `COMMAND-TRACKING.md`
2. **Analyze Current Implementation**: Study existing code in `src/`
3. **Plan Parameters**: Design named parameter structure
4. **Create Branch**: `git checkout -b refactor/command-name`

#### **STEP 2: IMPLEMENTATION**
1. **Create Parameter Parser**: Implement dual syntax support
2. **Update Command Function**: Modify to use named parameters
3. **Add camelCase Alias**: Register both old and new names
4. **Maintain Backward Compatibility**: Ensure old syntax still works

#### **STEP 3: TESTING**
1. **Create Test File**: `tests/refactored/command_name_test.tcl`
2. **Test Both Syntaxes**: Old positional + new named parameters
3. **Test Edge Cases**: Invalid parameters, missing values, etc.
4. **Performance Test**: Ensure no regression
5. **Run Full Test Suite**: `./run_tests.sh`

#### **STEP 4: DOCUMENTATION**
1. **Update API Documentation**: `docs/refactored/command_name.md`
2. **Create Migration Guide**: Show before/after examples
3. **Update Main Documentation**: Reference new syntax
4. **Add Code Examples**: Working examples for both syntaxes

#### **STEP 5: VALIDATION**
1. **Code Review**: Self-review implementation
2. **Test Validation**: All tests pass
3. **Documentation Review**: Examples work correctly
4. **Performance Check**: No significant overhead

#### **STEP 6: COMMIT & TRACK**
1. **Stage All Changes**: `git add .`
2. **Commit Changes**: `git commit -am "Refactor: command_name with named parameters"`
3. **Update Tracking**: Mark command complete in `COMMAND-TRACKING.md`
4. **Update Progress**: Recalculate completion percentage
5. **Push Branch**: `git push origin refactor/command-name`

---

## 📁 **FILE STRUCTURE FOR REFACTORING**

```
LIBTORCH/
├── src/
│   ├── libtorchtcl.cpp          # Main implementation
│   ├── parameter_parsing.cpp    # Dual syntax parser
│   └── parameter_parsing.h      # Parser headers
├── tests/
│   ├── refactored/              # New test files for refactored commands
│   │   ├── tensor_create_test.tcl
│   │   ├── zeros_test.tcl
│   │   └── ...
│   └── run_tests.sh            # Test runner
├── docs/
│   ├── refactored/              # Documentation for refactored commands
│   │   ├── tensor_create.md
│   │   ├── zeros.md
│   │   └── ...
│   ├── migration_guide.md       # Overall migration guide
│   └── api_reference.md         # Updated API reference
├── COMMAND-TRACKING.md          # Progress tracking
└── REFACTORING-WORKFLOW.md      # This workflow document
```

---

## 🔧 **IMPLEMENTATION TEMPLATES**

### **Parameter Parser Template**
```cpp
// In parameter_parsing.h
struct TensorCreationArgs {
    std::vector<int64_t> shape;
    std::string dtype = "float32";
    std::string device = "cpu";
    bool requires_grad = false;
    Tcl_Obj* data = nullptr;
    
    bool IsValid() const;
    static TensorCreationArgs Parse(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]);
};

// In parameter_parsing.cpp
TensorCreationArgs TensorCreationArgs::Parse(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    TensorCreationArgs args;
    
    if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
        // Positional syntax (backward compatibility)
        return ParsePositionalArgs(interp, objc, objv);
    } else {
        // Named parameter syntax
        return ParseNamedArgs(interp, objc, objv);
    }
}
```

### **Command Implementation Template**
```cpp
// In libtorchtcl.cpp
int TensorCreate_Cmd(ClientData clientData, Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
    try {
        // Parse arguments using dual syntax
        TensorCreationArgs args = TensorCreationArgs::Parse(interp, objc, objv);
        
        if (!args.IsValid()) {
            Tcl_SetResult(interp, (char*)"Invalid arguments for tensor_create", TCL_STATIC);
            return TCL_ERROR;
        }
        
        // Implementation using parsed arguments
        // ... existing implementation logic ...
        
        return TCL_OK;
    } catch (const std::exception& e) {
        Tcl_SetResult(interp, (char*)e.what(), TCL_STATIC);
        return TCL_ERROR;
    }
}

// Register both old and new names
Tcl_CreateObjCommand(interp, "torch::tensor_create", TensorCreate_Cmd, nullptr, nullptr);
Tcl_CreateObjCommand(interp, "torch::tensorCreate", TensorCreate_Cmd, nullptr, nullptr);
```

### **Test File Template**
```tcl
# tests/refactored/tensor_create_test.tcl
package require torch

# Test 1: Original positional syntax (backward compatibility)
puts "Testing original positional syntax..."
set tensor1 [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
puts "Original syntax: OK"

# Test 2: New named parameter syntax
puts "Testing new named parameter syntax..."
set tensor2 [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
puts "Named parameters: OK"

# Test 3: Mixed syntax (if supported)
puts "Testing mixed syntax..."
set tensor3 [torch::tensorCreate {1.0 2.0 3.0} -dtype float32 -device cpu]
puts "Mixed syntax: OK"

# Test 4: Error handling
puts "Testing error handling..."
if {[catch {torch::tensorCreate -data {1.0 2.0} -dtype invalid_type} result]} {
    puts "Error handling: OK - $result"
} else {
    puts "ERROR: Should have failed with invalid dtype"
}

# Test 5: Performance comparison
puts "Testing performance..."
set start [clock clicks -milliseconds]
for {set i 0} {$i < 1000} {incr i} {
    torch::tensor_create {1.0 2.0 3.0} float32 cpu true
}
set end [clock clicks -milliseconds]
puts "Performance: [expr {$end - $start}]ms for 1000 iterations"

puts "All tests passed for torch::tensor_create/torch::tensorCreate"
```

### **Documentation Template**
```markdown
# docs/refactored/tensor_create.md

# torch::tensor_create / torch::tensorCreate

## Overview
Creates a new tensor with specified data, dtype, device, and gradient requirements.

## Syntax

### Original Syntax (Backward Compatible)
```tcl
torch::tensor_create data dtype device requires_grad
```

### New Syntax (Named Parameters)
```tcl
torch::tensorCreate -data data -dtype dtype -device device -requiresGrad requires_grad
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-data` | list | required | Tensor data as a list of values |
| `-dtype` | string | "float32" | Data type (float32, float64, int32, etc.) |
| `-device` | string | "cpu" | Device placement (cpu, cuda) |
| `-requiresGrad` | boolean | false | Whether tensor requires gradients |

## Examples

### Basic Usage
```tcl
# Original syntax
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

# New syntax
set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
```

### With Defaults
```tcl
# Only specify data, use defaults for others
set tensor [torch::tensorCreate -data {1.0 2.0 3.0}]
```

### CUDA Tensor
```tcl
# Create tensor on GPU
set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -device cuda -requiresGrad true]
```

## Migration Guide

### Before (Old Syntax)
```tcl
torch::tensor_create {1.0 2.0 3.0} float32 cuda true
```

### After (New Syntax)
```tcl
torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cuda -requiresGrad true
```

## Notes
- Both syntaxes are supported for backward compatibility
- Named parameters provide better readability and flexibility
- Default values reduce parameter verbosity
- Error messages are more descriptive with named parameters
```

---

## 📋 **COMMAND PRIORITY ORDER**

### **Phase 1: Core Tensors (Start Here)**
1. `torch::zeros` - Most commonly used
2. `torch::ones` - Very common
3. `torch::tensor_create` - Core functionality
4. `torch::empty` - Common tensor creation
5. `torch::full` - Useful utility
6. `torch::tensor_add` - Basic arithmetic
7. `torch::tensor_mul` - Basic arithmetic
8. `torch::tensor_matmul` - Matrix operations
9. `torch::tensor_reshape` - Shape manipulation
10. `torch::tensor_cat` - Concatenation

### **Suggested Order for First 10 Commands**
```
Week 1: zeros, ones, tensor_create
Week 2: empty, full, tensor_add
Week 3: tensor_mul, tensor_matmul, tensor_reshape
Week 4: tensor_cat, tensor_sub
```

---

## 🧪 **TESTING CHECKLIST**

### **For Each Command, Test:**

#### **Functionality Tests**
- [ ] Original positional syntax works
- [ ] New named parameter syntax works
- [ ] Mixed syntax works (if supported)
- [ ] Default parameters work correctly
- [ ] All parameter combinations work

#### **Error Handling Tests**
- [ ] Invalid parameter names
- [ ] Missing required parameters
- [ ] Invalid parameter values
- [ ] Type mismatches
- [ ] Clear error messages

#### **Performance Tests**
- [ ] No significant performance regression
- [ ] Memory usage is reasonable
- [ ] Large tensor operations work
- [ ] CUDA operations work (if applicable)

#### **Integration Tests**
- [ ] Works with other refactored commands
- [ ] Works with existing unrefactored commands
- [ ] No breaking changes to existing workflows

---

## 📝 **DOCUMENTATION CHECKLIST**

### **For Each Command, Document:**

#### **API Documentation**
- [ ] Complete parameter list with types and defaults
- [ ] Both syntax examples (old and new)
- [ ] Clear parameter descriptions
- [ ] Return value documentation
- [ ] Error conditions and messages

#### **Examples**
- [ ] Basic usage examples
- [ ] Advanced usage examples
- [ ] Common patterns and workflows
- [ ] Migration examples (before/after)

#### **Migration Guide**
- [ ] Clear before/after comparison
- [ ] Parameter mapping explanation
- [ ] Common migration patterns
- [ ] Troubleshooting tips

---

## 🔄 **WORKFLOW AUTOMATION**

### **Create Helper Scripts**

#### **1. Command Selection Script**
```bash
#!/bin/bash
# scripts/select_next_command.sh
# Shows next command to refactor based on priority

echo "Next commands to refactor (in priority order):"
grep -n "\[ \]" COMMAND-TRACKING.md | head -10
```

#### **2. Test Runner Script**
```bash
#!/bin/bash
# scripts/test_refactored.sh command_name
# Runs tests for a specific refactored command

COMMAND=$1
if [ -z "$COMMAND" ]; then
    echo "Usage: $0 command_name"
    exit 1
fi

echo "Testing refactored command: $COMMAND"
./build.sh
tclsh tests/refactored/${COMMAND}_test.tcl
```

#### **3. Progress Tracker Script**
```bash
#!/bin/bash
# scripts/update_progress.sh
# Updates progress statistics

COMPLETED=$(grep -c "\[x\]" COMMAND-TRACKING.md)
TOTAL=489
PERCENTAGE=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc)

echo "Progress: $COMPLETED/$TOTAL commands completed ($PERCENTAGE%)"
```

---

## 🎯 **SUCCESS METRICS**

### **Per Command**
- [ ] ✅ Both syntaxes work correctly
- [ ] ✅ All tests pass
- [ ] ✅ Documentation is complete
- [ ] ✅ No performance regression
- [ ] ✅ Backward compatibility maintained

### **Overall Progress**
- [ ] ✅ 10% complete (49 commands)
- [ ] ✅ 25% complete (122 commands)
- [ ] ✅ 50% complete (245 commands)
- [ ] ✅ 75% complete (367 commands)
- [ ] ✅ 100% complete (489 commands)

---

## 🚀 **GETTING STARTED**

### **First Command: torch::zeros**

1. **Select**: Mark `torch::zeros` as next in `COMMAND-TRACKING.md`
2. **Implement**: Create parameter parser and update command
3. **Test**: Create `tests/refactored/zeros_test.tcl`
4. **Document**: Create `docs/refactored/zeros.md`
5. **Validate**: Run all tests and verify documentation
6. **Commit**: `git add . && git commit -am "Refactor: zeros with named parameters"`
7. **Track**: Mark complete and move to next command

### **Batch Refactoring (Multiple Commands)**
If refactoring multiple commands in one session:
1. **Complete all commands** in the batch
2. **Test all commands** individually
3. **Document all commands**
4. **Stage all changes**: `git add .`
5. **Commit batch**: `git commit -am "Refactor: batch of commands with named parameters (zeros, ones, tensor_create)"`
6. **Update tracking** for all commands in batch

### **Git Best Practices**
- **Always use `git add .`**: Ensures all changes are staged
- **Descriptive commit messages**: Include command names in commit message
- **Commit after each command**: Don't let changes accumulate
- **Use feature branches**: `git checkout -b refactor/command-name`
- **Push regularly**: `git push origin refactor/command-name`

### **Daily Workflow**
1. **Morning**: Select next command, plan implementation
2. **Implementation**: Code the refactoring
3. **Afternoon**: Create tests and documentation
4. **Evening**: Validate and commit changes
5. **Update**: Mark progress in tracking file
6. **Commit**: `git add . && git commit -am "Refactor: command_name with named parameters"`

---

**🎯 READY TO START SYSTEMATIC REFACTORING!**  
**📋 ONE COMMAND AT A TIME WITH FULL VALIDATION**  
**📊 COMPREHENSIVE TESTING AND DOCUMENTATION** 