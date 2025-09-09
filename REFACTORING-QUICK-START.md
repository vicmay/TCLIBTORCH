# LibTorch TCL Extension - Refactoring Quick Start Guide

**Status**: 🚀 **READY TO START** - Complete One-by-One Refactoring System  
**Approach**: Refactor → Test → Document → Commit → Repeat  
**Git Practice**: `git add .` + `git commit -am` after each command  

---

## 🎯 **COMPLETE WORKFLOW FOR EACH COMMAND**

### **Step 1: Select Next Command**
```bash
./scripts/select_next_command.sh
```
Shows next 10 commands in priority order with current progress.

### **Step 2: Create Feature Branch**
```bash
git checkout -b refactor/command-name
```

### **Step 3: Implement Refactoring**
- Update command in `src/libtorchtcl.cpp`
- Add parameter parser for named parameters
- Maintain backward compatibility
- Add camelCase alias if needed

### **Step 4: Create Test File**
```bash
cp tests/refactored/template_test.tcl tests/refactored/command_name_test.tcl
# Edit the test file for your specific command
```

### **Step 5: Test the Command**
```bash
./scripts/test_refactored.sh command_name
```
- Builds project
- Runs comprehensive tests
- Validates both syntaxes
- Shows next steps

### **Step 6: Create Documentation**
```bash
cp docs/refactored/template.md docs/refactored/command_name.md
# Edit the documentation for your specific command
```

### **Step 7: Commit Changes**
```bash
./scripts/commit_refactored.sh command_name
```
- Stages all changes with `git add .`
- Commits with descriptive message
- Shows progress update

### **Step 8: Update Tracking**
- Mark command complete in `COMMAND-TRACKING.md`
- Check progress: `./scripts/update_progress.sh`

---

## 📊 **PROGRESS MONITORING**

### **Check Current Progress**
```bash
./scripts/update_progress.sh
```
Shows:
- Overall completion percentage
- Progress by phase
- Remaining work estimate
- Recent completed commands

### **View Next Commands**
```bash
./scripts/select_next_command.sh
```
Shows next 10 commands to work on.

---

## 🔄 **BATCH REFACTORING**

If refactoring multiple commands in one session:

### **Complete All Commands**
- Implement all commands
- Test each individually
- Document each command

### **Commit Batch**
```bash
./scripts/commit_refactored.sh batch 'command1,command2,command3'
```
- Stages all changes with `git add .`
- Commits with batch message
- Updates progress

---

## 📁 **FILE STRUCTURE**

```
LIBTORCH/
├── COMMAND-TRACKING.md          # All 489 commands with checkboxes
├── REFACTORING-WORKFLOW.md      # Detailed workflow guide
├── REFACTORING-QUICK-START.md   # This quick start guide
├── scripts/
│   ├── select_next_command.sh   # Shows next commands
│   ├── test_refactored.sh       # Tests specific commands
│   ├── commit_refactored.sh     # Commits with git add .
│   └── update_progress.sh       # Shows progress statistics
├── tests/refactored/
│   └── template_test.tcl        # Test file template
└── docs/refactored/
    └── template.md              # Documentation template
```

---

## 🎯 **RECOMMENDED STARTING POINTS**

### **Phase 1 Priority Order (Core Tensors)**
1. `torch::zeros` - Most commonly used
2. `torch::ones` - Very common
3. `torch::tensor_create` - Core functionality
4. `torch::empty` - Common tensor creation
5. `torch::full` - Useful utility

### **Example: Refactoring torch::zeros**
```bash
# 1. Select next command
./scripts/select_next_command.sh

# 2. Create branch
git checkout -b refactor/zeros

# 3. Implement refactoring
# ... edit src/libtorchtcl.cpp ...

# 4. Create test
cp tests/refactored/template_test.tcl tests/refactored/zeros_test.tcl
# ... edit test file ...

# 5. Test
./scripts/test_refactored.sh zeros

# 6. Create documentation
cp docs/refactored/template.md docs/refactored/zeros.md
# ... edit documentation ...

# 7. Commit
./scripts/commit_refactored.sh zeros

# 8. Update tracking
# ... mark complete in COMMAND-TRACKING.md ...
```

---

## 📋 **SUCCESS CHECKLIST**

### **For Each Command**
- [ ] ✅ Both old and new syntax work
- [ ] ✅ All tests pass
- [ ] ✅ Documentation is complete
- [ ] ✅ No performance regression
- [ ] ✅ Backward compatibility maintained
- [ ] ✅ Changes committed with `git add .`

### **Git Best Practices**
- [ ] ✅ Always use `git add .` (not individual files)
- [ ] ✅ Use descriptive commit messages
- [ ] ✅ Commit after each command (don't accumulate)
- [ ] ✅ Use feature branches
- [ ] ✅ Push regularly

---

## 🚀 **GETTING STARTED RIGHT NOW**

### **1. Check Current Status**
```bash
./scripts/update_progress.sh
```

### **2. See Next Commands**
```bash
./scripts/select_next_command.sh
```

### **3. Start with First Command**
```bash
git checkout -b refactor/zeros
# ... implement, test, document, commit ...
```

---

## 📈 **PROGRESS TARGETS**

### **Weekly Goals**
- **Week 1**: Infrastructure + 5 commands
- **Week 2**: 10 commands (Phase 1 core)
- **Week 3-4**: 20 commands (Phase 1 complete)
- **Week 5-6**: 30 commands (Phase 2 start)

### **Success Metrics**
- **10% complete**: 49 commands (Week 5)
- **25% complete**: 122 commands (Week 12)
- **50% complete**: 245 commands (Week 24)
- **100% complete**: 489 commands (Week 48)

---

## 💡 **TROUBLESHOOTING**

### **Test Fails**
1. Check test file syntax
2. Verify command implementation
3. Check for compilation errors
4. Run individual test cases

### **Commit Fails**
1. Ensure you're in git repository
2. Check for uncommitted changes
3. Verify branch name is correct
4. Check git status

### **Progress Not Updating**
1. Verify checkboxes are marked `[x]` in `COMMAND-TRACKING.md`
2. Run `./scripts/update_progress.sh`
3. Check file syntax

---

**🎯 READY TO START REFACTORING!**  
**📋 SYSTEMATIC APPROACH WITH GIT BEST PRACTICES**  
**📊 COMPREHENSIVE PROGRESS TRACKING**  
**🚀 ONE COMMAND AT A TIME WITH FULL VALIDATION** 