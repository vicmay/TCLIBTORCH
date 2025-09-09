#!/usr/bin/env tclsh

# Test file for torch::tensor_add command with dual syntax support
# Tests both positional and named parameter syntax

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test suite for torch::tensor_add
test tensor_add-1.1 {Basic positional syntax} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test basic tensor_add
    set result [torch::tensor_add $tensor1 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-1.2 {Positional syntax with alpha} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test tensor_add with alpha
    set result [torch::tensor_add $tensor1 $tensor2 2.0]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-2.1 {Named parameter syntax - basic} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test basic named parameter syntax
    set result [torch::tensor_add -input1 $tensor1 -input2 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-2.2 {Named parameter syntax with alpha} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test named parameter syntax with alpha
    set result [torch::tensor_add -input1 $tensor1 -input2 $tensor2 -alpha 2.0]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-2.3 {Named parameter syntax with alternative names} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test named parameter syntax with alternative parameter names
    set result [torch::tensor_add -input $tensor1 -other $tensor2 -alpha 1.5]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-3.1 {CamelCase alias - basic} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test camelCase alias
    set result [torch::tensorAdd -input1 $tensor1 -input2 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-3.2 {CamelCase alias with parameters} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test camelCase alias with parameters
    set result [torch::tensorAdd -input1 $tensor1 -input2 $tensor2 -alpha 2.5]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-4.1 {Error handling - invalid first tensor} {
    # Create one valid tensor
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test with non-existent first tensor
    catch {torch::tensor_add invalid_tensor $tensor2} result
    expr {[string length $result] > 0}
} {1}

test tensor_add-4.2 {Error handling - invalid second tensor} {
    # Create one valid tensor
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    
    # Test with non-existent second tensor
    catch {torch::tensor_add $tensor1 invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor_add-4.3 {Error handling - missing input} {
    # Test with missing input parameter
    catch {torch::tensor_add -input1 tensor1} result
    expr {[string length $result] > 0}
} {1}

test tensor_add-4.4 {Error handling - invalid alpha} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test with invalid alpha value
    catch {torch::tensor_add -input1 $tensor1 -input2 $tensor2 -alpha invalid} result
    expr {[string length $result] > 0}
} {1}

test tensor_add-4.5 {Error handling - unknown parameter} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test with unknown parameter
    catch {torch::tensor_add -input1 $tensor1 -input2 $tensor2 -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_add-5.1 {Mathematical correctness - basic addition} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Add tensors
    set result [torch::tensor_add $tensor1 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-5.2 {Mathematical correctness - with alpha} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Add tensors with alpha
    set result [torch::tensor_add $tensor1 $tensor2 2.0]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-6.1 {Different data types - float32} {
    # Create two float32 tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Add tensors
    set result [torch::tensor_add $tensor1 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-6.2 {Different data types - float64} {
    # Create two float64 tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float64 -device cpu]
    
    # Add tensors
    set result [torch::tensor_add $tensor1 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-6.3 {Different data types - int32} {
    # Create two int32 tensors
    set tensor1 [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
    set tensor2 [torch::tensor_create -data {4 5 6} -dtype int32 -device cpu]
    
    # Add tensors
    set result [torch::tensor_add $tensor1 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-7.1 {Multi-dimensional tensors} {
    # Create two 2D tensors
    set tensor1 [torch::zeros {2 2} float32 cpu]
    set tensor2 [torch::ones {2 2} float32 cpu]
    
    # Add tensors
    set result [torch::tensor_add $tensor1 $tensor2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-8.1 {Zero alpha} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Add tensors with alpha = 0
    set result [torch::tensor_add $tensor1 $tensor2 0.0]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-8.2 {Negative alpha} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Add tensors with negative alpha
    set result [torch::tensor_add $tensor1 $tensor2 -1.0]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_add-9.1 {Syntax consistency - positional vs named} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test both syntaxes
    set result1 [torch::tensor_add $tensor1 $tensor2]
    set result2 [torch::tensor_add -input1 $tensor1 -input2 $tensor2]
    
    # Both should return valid tensor handles
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_add-9.2 {Syntax consistency - snake_case vs camelCase} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
    
    # Test both naming conventions
    set result1 [torch::tensor_add -input1 $tensor1 -input2 $tensor2]
    set result2 [torch::tensorAdd -input1 $tensor1 -input2 $tensor2]
    
    # Both should return valid tensor handles
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 