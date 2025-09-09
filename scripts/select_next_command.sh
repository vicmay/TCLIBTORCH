#!/bin/bash
# scripts/select_next_command.sh
# Automatically finds next command to refactor by checking actual implementation

echo "=== LibTorch TCL Extension - Next Commands to Refactor ==="
echo ""

# Function to check if a command has dual syntax support
has_dual_syntax() {
    local cmd=$1
    local snake_name=$(echo "$cmd" | sed 's/torch:://')
    
    # Handle the naming pattern correctly
    local func_name
    if [[ "$snake_name" == tensor_* ]]; then
        func_name=$(echo "$snake_name" | sed -r 's/(^|_)([a-z])/\U\2/g')_Cmd
    else
        func_name="Tensor$(echo "$snake_name" | sed -r 's/(^|_)([a-z])/\U\2/g')_Cmd"
    fi
    
    # Check if the command function has dual syntax support
    # Look for the inline dual syntax pattern
    if grep -A 20 "int ${func_name}" src/*.cpp 2>/dev/null | grep -q "use_named_params"; then
        return 0
    fi
    
    if grep -A 20 "int ${func_name}" src/*.cpp 2>/dev/null | grep -q "starts with -"; then
        return 0
    fi
    
    # Check for TensorCreationArgs::Parse pattern
    if grep -A 10 "int ${func_name}" src/*.cpp 2>/dev/null | grep -q "TensorCreationArgs.*Parse"; then
        return 0
    fi
    
    # Check for other dual syntax patterns (fix the escaping)
    if grep -A 20 "int ${func_name}" src/*.cpp 2>/dev/null | grep -q "namedSyntax\|named.*syntax"; then
        return 0
    fi
    
    # Check for the objv[1][0] == '-' pattern with proper escaping
    if grep -A 20 "int ${func_name}" src/*.cpp 2>/dev/null | grep -q "objv\[1\]\[0\] == '-'"; then
        return 0
    fi
    
    # Check for the Tcl_GetString(objv[1])[0] == '-' pattern
    if grep -A 20 "int ${func_name}" src/*.cpp 2>/dev/null | grep -q "Tcl_GetString(objv\[1\])\[0\] == '-'"; then
        return 0
    fi
    
    # Also check for the ParseXXXArgs pattern (some commands use this)
    local func_base=$(echo "$snake_name" | sed -r 's/(^|_)([a-z])/\U\2/g')
    if grep -qi "Parse${func_base}Args\|struct ${func_base}Args\|Parse${func_base}TensorArgs\|struct ${func_base}TensorArgs" src/*.cpp 2>/dev/null; then
        return 0
    fi
    
    return 1
}

# Function to check if a command has camelCase alias
has_camel_case_alias() {
    local cmd=$1
    local snake_name=$(echo "$cmd" | sed 's/torch:://')
    
    # Convert snake_case to camelCase
    local camel_name=$(echo "$snake_name" | sed -r 's/_([a-z])/\U\1/g')
    
    # Check if both snake_case and camelCase are registered
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

# List of all commands to check (core tensor operations first)
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
    
    # Phase 2: Neural Network Layers
    "torch::linear"
    "torch::conv1d"
    "torch::conv2d"
    "torch::conv3d"
    "torch::maxpool1d"
    "torch::maxpool2d"
    "torch::maxpool3d"
    "torch::avgpool1d"
    "torch::avgpool2d"
    "torch::avgpool3d"
    "torch::dropout"
    "torch::batchnorm2d"
    "torch::batch_norm1d"
    "torch::batch_norm3d"
    "torch::layer_norm"
    "torch::group_norm"
    "torch::lstm"
    "torch::gru"
    "torch::rnn_tanh"
    "torch::rnn_relu"
)

# Count completed and find next to refactor
completed=0
next_commands=()

echo "🔍 Scanning codebase for refactoring status..."
echo ""

for cmd in "${COMMANDS[@]}"; do
    if is_fully_refactored "$cmd"; then
        ((completed++))
    else
        if [ ${#next_commands[@]} -lt 10 ]; then
            next_commands+=("$cmd")
        fi
    fi
done

total=${#COMMANDS[@]}
percentage=$(echo "scale=1; $completed * 100 / $total" | bc 2>/dev/null || echo "0")

echo "📊 Progress: $completed/$total commands completed ($percentage%)"
echo ""

if [ ${#next_commands[@]} -eq 0 ]; then
    echo "🎉 All tracked commands are fully refactored!"
    echo ""
    echo "Consider adding more commands to the list or checking other areas:"
    echo "- Optimizers and schedulers"
    echo "- Loss functions"
    echo "- Advanced operations"
else
    echo "🎯 Next ${#next_commands[@]} commands to refactor:"
    echo ""
    
    for i in "${!next_commands[@]}"; do
        cmd="${next_commands[$i]}"
        echo "  $((i+1)). $cmd"
        
        # Show what's missing
        missing=""
        if ! has_dual_syntax "$cmd"; then
            missing="$missing dual-syntax"
        fi
        if ! has_camel_case_alias "$cmd"; then
            missing="$missing camelCase-alias"
        fi
        if ! has_tests "$cmd"; then
            missing="$missing tests"
        fi
        
        if [ -n "$missing" ]; then
            echo "      Missing:$missing"
        fi
        echo ""
    done
fi

echo "💡 To refactor a command:"
echo "   1. Implement dual syntax parser (ParseXXXArgs function)"
echo "   2. Register both snake_case and camelCase aliases"
echo "   3. Create comprehensive tests"
echo "   4. Update documentation"
echo ""
echo "🔧 Build and test: cd build && make -j4 && cd ../tests/refactored && tclsh [command]_test.tcl" 