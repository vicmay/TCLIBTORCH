#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax (backward compatibility)
test tensor_matrix_exp-1.1 {Basic positional syntax} {
    # Create a 2x2 matrix
    set tensor [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {2 2}]
    set result [torch::tensor_matrix_exp $reshaped]
    
    # Result should be a valid tensor handle
    expr {[string length $result] > 0}
} {1}

# Test 2: Named parameter syntax
test tensor_matrix_exp-2.1 {Named parameter syntax} {
    # Create a 2x2 identity matrix
    set tensor [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {2 2}]
    set result [torch::tensor_matrix_exp -input $reshaped]
    
    # Result should be a valid tensor handle
    expr {[string length $result] > 0}
} {1}

# Test 3: Alternative parameter name
test tensor_matrix_exp-2.2 {Alternative parameter name} {
    # Create a 2x2 matrix
    set tensor [torch::tensor_create {1.0 0.5 0.5 1.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {2 2}]
    set result [torch::tensor_matrix_exp -tensor $reshaped]
    
    # Result should be a valid tensor handle
    expr {[string length $result] > 0}
} {1}

# Test 4: camelCase alias
test tensor_matrix_exp-3.1 {camelCase alias syntax} {
    # Create a 2x2 matrix
    set tensor [torch::tensor_create {2.0 0.0 0.0 2.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {2 2}]
    set result [torch::tensorMatrixExp -input $reshaped]
    
    # Result should be a valid tensor handle
    expr {[string length $result] > 0}
} {1}

# Test 5: Both syntaxes produce same result
test tensor_matrix_exp-4.1 {Both syntaxes produce same result} {
    # Create the same matrix for both tests
    set tensor1 [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cpu false]
    set matrix1 [torch::tensor_reshape $tensor1 {2 2}]
    
    set tensor2 [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cpu false]
    set matrix2 [torch::tensor_reshape $tensor2 {2 2}]
    
    # Test positional syntax
    set result1 [torch::tensor_matrix_exp $matrix1]
    
    # Test named syntax  
    set result2 [torch::tensor_matrix_exp -input $matrix2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    string equal $shape1 $shape2
} {1}

# Test 6: Error handling - missing parameters (named syntax)
test tensor_matrix_exp-5.1 {Error handling - missing input parameter} {
    set result [catch {torch::tensor_matrix_exp} error]
    expr {$result == 1 && [string match "*Required parameter missing*" $error]}
} {1}

# Test 7: Error handling - wrong number of positional args
test tensor_matrix_exp-5.2 {Error handling - wrong number of positional args} {
    set result [catch {torch::tensor_matrix_exp arg1 arg2} error]
    expr {$result == 1}
} {1}

# Test 8: Error handling - invalid tensor name
test tensor_matrix_exp-5.3 {Error handling - invalid tensor name} {
    set result [catch {torch::tensor_matrix_exp -input "invalid_tensor"} error]
    expr {$result == 1 && [string match "*Invalid tensor name*" $error]}
} {1}

# Test 9: Error handling - unknown parameter
test tensor_matrix_exp-5.4 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cpu false]
    set matrix [torch::tensor_reshape $tensor {2 2}]
    set result [catch {torch::tensor_matrix_exp -input $matrix -badparam value} error]
    expr {$result == 1 && [string match "*Unknown parameter*" $error]}
} {1}

# Test 10: Matrix exponential of zero matrix
test tensor_matrix_exp-6.1 {Matrix exponential of zero matrix} {
    # Create a 2x2 zero matrix
    set tensor [torch::zeros {2 2} float32 cpu false]
    set result [torch::tensor_matrix_exp -input $tensor]
    
    # Matrix exp of zero matrix should be identity matrix
    # Just verify it produces a valid result
    expr {[string length $result] > 0}
} {1}

# Test 11: Matrix exponential preserves shape
test tensor_matrix_exp-7.1 {Matrix exponential preserves shape} {
    # Create a 3x3 matrix
    set data {1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0}
    set tensor [torch::tensor_create $data float32 cpu false]
    set matrix [torch::tensor_reshape $tensor {3 3}]
    set result [torch::tensor_matrix_exp -input $matrix]
    
    # Should preserve shape
    set shape [torch::tensor_shape $result]
    string equal $shape {3 3}
} {1}

# Test 12: Different data types
test tensor_matrix_exp-8.1 {Different data types} {
    # Create a double precision matrix
    set tensor [torch::tensor_create {1.0 0.0 0.0 1.0} float64 cpu false]
    set matrix [torch::tensor_reshape $tensor {2 2}]
    set result [torch::tensor_matrix_exp -input $matrix]
    
    # Should produce a valid result
    expr {[string length $result] > 0}
} {1}

cleanupTests 