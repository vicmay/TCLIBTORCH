#!/bin/bash
# scripts/update_progress.sh
# Automatically updates COMMAND-TRACKING.md based on actual codebase status

echo "=== Updating Command Tracking Based on Codebase ==="
echo ""

# Function to check if a command has dual syntax support
has_dual_syntax() {
    local cmd=$1
    local snake_name=$(echo "$cmd" | sed 's/torch:://')
    local func_base=$(echo "$snake_name" | sed -r 's/(^|_)([a-z])/\U\2/g')
    
    if grep -q "Parse${func_base}Args\|Parse.*${func_base}.*Args\|struct ${func_base}Args\|${func_base}Args Parse" src/*.cpp 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to check if a command has camelCase alias
has_camel_case_alias() {
    local cmd=$1
    local snake_name=$(echo "$cmd" | sed 's/torch:://')
    local camel_name=$(echo "$snake_name" | sed -r 's/_([a-z])/\U\1/g')
    
    if grep -q "\"torch::$snake_name\"" src/libtorchtcl.cpp 2>/dev/null && \
       grep -q "\"torch::$camel_name\"" src/libtorchtcl.cpp 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to check if a command has tests
has_tests() {
    local cmd=$1
    local snake_name=$(echo "$cmd" | sed 's/torch:://')
    local test_file="tests/refactored/${snake_name}_test.tcl"
    
    if [ -f "$test_file" ]; then
        return 0
    fi
    return 1
}

# Function to check if a command is fully refactored
is_fully_refactored() {
    local cmd=$1
    
    if has_dual_syntax "$cmd" && has_camel_case_alias "$cmd" && has_tests "$cmd"; then
        return 0
    fi
    return 1
}

# Create a temporary file for the new tracking content
TEMP_FILE=$(mktemp)

# List of commands to check (same as in select_next_command.sh)
COMMANDS=(
    # Phase 1.1: Tensor Creation
    "torch::arange"
    "torch::empty"
    "torch::empty_like"
    "torch::eye"
    "torch::full"
    "torch::full_like"
    "torch::linspace"
    "torch::logspace"
    "torch::ones"
    "torch::ones_like"
    "torch::zeros"
    "torch::zeros_like"
    "torch::rand_like"
    "torch::randn_like"
    "torch::randint_like"
    "torch::tensor_create"
    
    # Phase 1.2: Basic Operations
    "torch::tensor_abs"
    "torch::tensor_add"
    "torch::tensor_sub"
    "torch::tensor_mul"
    "torch::tensor_div"
    "torch::tensor_matmul"
    "torch::tensor_bmm"
    "torch::tensor_exp"
    "torch::tensor_log"
    "torch::tensor_sqrt"
    "torch::tensor_sigmoid"
    "torch::tensor_relu"
    "torch::tensor_tanh"
    "torch::tensor_sum"
    "torch::tensor_mean"
    "torch::tensor_max"
    "torch::tensor_min"
    
    # Phase 1.3: Properties & Device Operations
    "torch::tensor_dtype"
    "torch::tensor_device"
    "torch::tensor_requires_grad"
    "torch::tensor_grad"
    "torch::tensor_to"
    "torch::tensor_backward"
    "torch::tensor_shape"
    "torch::tensor_item"
    "torch::tensor_numel"
    "torch::tensor_print"
    "torch::tensor_rand"
    "torch::tensor_randn"
    "torch::tensor_cat"
    "torch::tensor_stack"
    "torch::tensor_reshape"
    "torch::tensor_permute"
)

# Count completed commands
completed=0
total=${#COMMANDS[@]}

echo "🔍 Scanning $total commands..."

# Generate new tracking file content
cat > "$TEMP_FILE" << 'EOF'
# LibTorch TCL Extension - Command Refactoring Tracking (Auto-Generated)

**Last Updated**: $(date)
**Auto-Generated**: This file is automatically updated based on codebase analysis

---

## 📊 **PROGRESS SUMMARY**

EOF

# Count completed commands first
for cmd in "${COMMANDS[@]}"; do
    if is_fully_refactored "$cmd"; then
        ((completed++))
    fi
done

percentage=$(echo "scale=1; $completed * 100 / $total" | bc 2>/dev/null || echo "0")

cat >> "$TEMP_FILE" << EOF
- **Total Commands Tracked**: $total
- **Completed**: $completed ($percentage%)
- **Remaining**: $((total - completed))

## 🎯 **REFACTORING STATUS**

### **Legend**
- ✅ **Fully Refactored**: Dual syntax + camelCase alias + tests
- 🔄 **In Progress**: Some components implemented
- ❌ **Not Started**: Missing most/all components

---

## 📋 **DETAILED COMMAND STATUS**

### **Phase 1.1: Tensor Creation Commands**

EOF

# Process each command and add to tracking file
current_phase=""
for cmd in "${COMMANDS[@]}"; do
    # Determine phase
    case "$cmd" in
        "torch::arange"|"torch::empty"|"torch::empty_like"|"torch::eye"|"torch::full"|"torch::full_like"|"torch::linspace"|"torch::logspace"|"torch::ones"|"torch::ones_like"|"torch::zeros"|"torch::zeros_like"|"torch::rand_like"|"torch::randn_like"|"torch::randint_like"|"torch::tensor_create")
            phase="Phase 1.1: Tensor Creation Commands"
            ;;
        "torch::tensor_abs"|"torch::tensor_add"|"torch::tensor_sub"|"torch::tensor_mul"|"torch::tensor_div"|"torch::tensor_matmul"|"torch::tensor_bmm"|"torch::tensor_exp"|"torch::tensor_log"|"torch::tensor_sqrt"|"torch::tensor_sigmoid"|"torch::tensor_relu"|"torch::tensor_tanh"|"torch::tensor_sum"|"torch::tensor_mean"|"torch::tensor_max"|"torch::tensor_min")
            phase="Phase 1.2: Basic Tensor Operations"
            ;;
        *)
            phase="Phase 1.3: Properties & Device Operations"
            ;;
    esac
    
    # Add phase header if changed
    if [ "$current_phase" != "$phase" ]; then
        if [ -n "$current_phase" ]; then
            echo "" >> "$TEMP_FILE"
        fi
        echo "### **$phase**" >> "$TEMP_FILE"
        echo "" >> "$TEMP_FILE"
        current_phase="$phase"
    fi
    
    # Check status
    if is_fully_refactored "$cmd"; then
        echo "- [x] $cmd ✅" >> "$TEMP_FILE"
        echo "  - Dual syntax: ✅" >> "$TEMP_FILE"
        echo "  - camelCase alias: ✅" >> "$TEMP_FILE"
        echo "  - Tests: ✅" >> "$TEMP_FILE"
    else
        echo "- [ ] $cmd" >> "$TEMP_FILE"
        
        # Show what's missing
        if has_dual_syntax "$cmd"; then
            echo "  - Dual syntax: ✅" >> "$TEMP_FILE"
        else
            echo "  - Dual syntax: ❌" >> "$TEMP_FILE"
        fi
        
        if has_camel_case_alias "$cmd"; then
            echo "  - camelCase alias: ✅" >> "$TEMP_FILE"
        else
            echo "  - camelCase alias: ❌" >> "$TEMP_FILE"
        fi
        
        if has_tests "$cmd"; then
            echo "  - Tests: ✅" >> "$TEMP_FILE"
        else
            echo "  - Tests: ❌" >> "$TEMP_FILE"
        fi
    fi
    echo "" >> "$TEMP_FILE"
done

# Add footer
cat >> "$TEMP_FILE" << 'EOF'

---

## 🚀 **NEXT STEPS**

Run `./scripts/select_next_command.sh` to see the next commands to refactor.

For each incomplete command:
1. Implement dual syntax parser (ParseXXXArgs function)
2. Register both snake_case and camelCase aliases in src/libtorchtcl.cpp
3. Create comprehensive tests in tests/refactored/
4. Update documentation

EOF

# Replace the original file
mv "$TEMP_FILE" COMMAND-TRACKING.md

echo "✅ Updated COMMAND-TRACKING.md with current codebase status"
echo "📊 Progress: $completed/$total commands completed ($percentage%)"
echo ""
echo "Run './scripts/select_next_command.sh' to see next commands to refactor." 