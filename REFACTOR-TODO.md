# LibTorch TCL Extension - Complete API Refactoring Plan

**Status**: 🚀 **ACTIVE REFACTORING** - Major API Modernization Initiative  
**Target**: Convert 588 commands to named parameters + camelCase syntax  
**Progress**: 445/588 commands complete (75.7%)  
**Timeline**: Final phase - completing remaining commands  
**Compatibility**: Maintaining 100% backward compatibility  
**Tracking**: SQLite database for accurate real-time progress monitoring  

---

## 🎯 **REFACTORING OBJECTIVES**

### **Primary Goals**
1. **Named Parameters**: Convert from positional to `-option value` syntax
2. **camelCase Conversion**: Modernize from `snake_case` to `camelCase`
3. **Backward Compatibility**: Support both old and new syntaxes simultaneously
4. **Documentation**: Update all documentation and examples
5. **Testing**: Comprehensive validation of both syntaxes

### **Current Progress** (SQLite-Tracked)
- Commands with dual syntax: 509 (86.6%)
- Commands with camelCase alias: 492 (83.7%)
- Commands with tests: 457 (77.7%)
- Commands with documentation: 456 (77.6%)

### **Example Transformation**
```tcl
# BEFORE (Current - snake_case + positional)
torch::tensor_create {1.0 2.0 3.0} float32 cuda true
torch::lstm 128 256 2 true false 0.1 false

# AFTER (New - camelCase + named parameters)
torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cuda -requiresGrad true
torch::lstm -inputSize 128 -hiddenSize 256 -numLayers 2 -bias true -dropout 0.1

# BOTH SUPPORTED (Backward compatibility)
torch::tensor_create {1.0 2.0 3.0} float32 cuda true    # Still works
torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cuda -requiresGrad true  # New syntax
```

---

## 🗄️ **SQLITE TRACKING SYSTEM**

### **Key Features**
- **Automatic Code Scanning**: Database automatically scans source code to detect current implementation status
- **Real-time Accuracy**: Live tracking of dual syntax, camelCase aliases, tests, and documentation
- **Command Discovery**: Currently tracking 588 total commands
- **Category Breakdown**: Commands organized by logical categories with priority ordering
- **Search & Query**: Powerful filtering and search capabilities
- **Progress Analytics**: Detailed statistics and completion tracking

### **Database Tools**

#### **Query Next Commands**
```bash
# Show next commands to refactor (default: 10)
python3 scripts/query_next_commands.py

# Show next 5 commands with detailed status
python3 scripts/query_next_commands.py --next 5

# Show progress by category
python3 scripts/query_next_commands.py --category

# Search for specific commands
python3 scripts/query_next_commands.py --search tensor

# Show all incomplete commands
python3 scripts/query_next_commands.py --all
```

#### **Update Command Status**
```bash
# Mark command as completely refactored
python3 scripts/update_command_status.py mark-complete torch::command_name

# Update specific aspects
python3 scripts/update_command_status.py update torch::command_name --dual-syntax true --tests true

# Rescan specific command from codebase
python3 scripts/update_command_status.py rescan torch::command_name

# Rescan all commands (full update)
python3 scripts/update_command_status.py rescan-all
```

### **Next Commands to Refactor**

The following commands are currently prioritized for refactoring:

1. **torch::ne**
   - Category: other
   - Status: Needs dual syntax, tests, and documentation
   - Has camelCase alias

2. **torch::nms**
   - Category: other
   - Status: Needs dual syntax, tests, and documentation
   - Has camelCase alias

3. **torch::no_grad**
   - Category: other
   - Status: Needs dual syntax, camelCase alias, tests, and documentation

4. **torch::normal**
   - Category: other
   - Status: Has dual syntax and camelCase alias
   - Needs tests and documentation

5. **torch::normalize_image**
   - Category: other
   - Status: Needs all implementations (dual syntax, camelCase, tests, docs)

6. **torch::outer**
   - Category: other
   - Status: Needs dual syntax, tests, and documentation
   - Has camelCase alias

7. **torch::parameters_to**
   - Category: other
   - Status: Needs all implementations

8. **torch::pitch_shift**
   - Category: other
   - Status: Needs all implementations

9. **torch::pixel_shuffle**
   - Category: other
   - Status: Needs all implementations

10. **torch::pixel_unshuffle**
    - Category: other
    - Status: Needs all implementations

### **Implementation Pattern**

For each remaining command:

1. **Dual Syntax Implementation**
   ```cpp
   CommandArgs ParseDualSyntax(Tcl_Interp* interp, int objc, Tcl_Obj* const objv[]) {
       if (objc >= 2 && Tcl_GetString(objv[1])[0] != '-') {
           // Positional syntax (backward compatibility)
           return ParsePositionalArgs(interp, objc, objv);
       } else {
           // Named parameter syntax
           return ParseNamedArgs(interp, objc, objv);
       }
   }
   ```

2. **Test File Creation**
   ```tcl
   # In tests/refactored/command_test.tcl
   test command-1.1 {Basic positional syntax} {
       # Test implementation
   } {expected_result}

   test command-2.1 {Named parameter syntax} {
       # Test implementation
   } {expected_result}
   ```

3. **Documentation Update**
   - Create `docs/refactored/command.md`
   - Document both syntaxes
   - Include migration examples

4. **Command Registration**
   ```cpp
   // In src/libtorchtcl.cpp
   Tcl_CreateObjCommand(interp, "torch::command_name", CommandName_Cmd, NULL, NULL);
   Tcl_CreateObjCommand(interp, "torch::commandName", CommandName_Cmd, NULL, NULL);
   ```

---

## 📊 **PROGRESS TRACKING**

### **Current Statistics**
```
Total Commands: 588
Complete Commands: 445 (75.7%)
Commands with dual syntax: 509 (86.6%)
Commands with camelCase alias: 492 (83.7%)
Commands with tests: 457 (77.7%)
Commands with documentation: 456 (77.6%)
```

### **Success Criteria**
- [ ] ✅ All 588 commands support named parameters (509/588 complete)
- [ ] ✅ All commands available in camelCase (492/588 have aliases)
- [ ] ✅ 100% backward compatibility (maintained)
- [ ] ✅ Zero performance regression (validated)
- [ ] ✅ Complete documentation update (456/588 documented)

### **Workflow**
1. **Check next command**: `python3 scripts/query_next_commands.py --next 1`
2. **Implement refactoring**: Follow established patterns
3. **Mark complete**: `python3 scripts/update_command_status.py mark-complete torch::command_name`
4. **Commit changes**: Use git as normal
5. **Verify accuracy**: Database automatically updates on next scan

---

**🎯 TARGET: Complete remaining 143 commands**  
**🚀 IMPACT: Revolutionary API improvement**  
**📈 RESULT: Modern, professional, joy-to-use LibTorch TCL Extension**  
**🗄️ POWERED BY: SQLite real-time tracking system**
