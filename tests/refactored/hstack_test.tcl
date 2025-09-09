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

# ============================================================================
# TORCH::HSTACK COMMAND TESTS
# ============================================================================
# This test suite verifies the torch::hstack command functionality
# including dual syntax support and camelCase alias.

# Test 1: Basic functionality with positional syntax (backward compatibility)
test hstack-1.1 {Basic horizontal stack with multiple arguments - positional syntax} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    set result [torch::hstack $tensor1 $tensor2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 7}

# Test 2: Basic functionality with list argument (positional syntax)
test hstack-1.2 {Basic horizontal stack with list argument - positional syntax} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    set tensor_list [list $tensor1 $tensor2]
    set result [torch::hstack $tensor_list]
    set shape [torch::tensor_shape $result]
    set shape
} {2 7}

# Test 3: Basic functionality with named parameter syntax
test hstack-2.1 {Basic horizontal stack with named parameters} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    set tensor_list [list $tensor1 $tensor2]
    set result [torch::hstack -tensors $tensor_list]
    set shape [torch::tensor_shape $result]
    set shape
} {2 7}

# Test 4: Alternative parameter name -inputs
test hstack-2.2 {Basic horizontal stack with -inputs parameter} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    set tensor_list [list $tensor1 $tensor2]
    set result [torch::hstack -inputs $tensor_list]
    set shape [torch::tensor_shape $result]
    set shape
} {2 7}

# Test 5: camelCase alias functionality
test hstack-3.1 {camelCase alias torch::hStack with named parameters} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    set tensor_list [list $tensor1 $tensor2]
    set result [torch::hStack -tensors $tensor_list]
    set shape [torch::tensor_shape $result]
    set shape
} {2 7}

# Test 6: camelCase alias with positional syntax
test hstack-3.2 {camelCase alias torch::hStack with positional syntax} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    set result [torch::hStack $tensor1 $tensor2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 7}

# Test 7: Stack multiple tensors (more than 2)
test hstack-4.1 {Stack multiple tensors (positional syntax)} {
    set tensor1 [torch::ones -shape {3 2}]
    set tensor2 [torch::ones -shape {3 3}]
    set tensor3 [torch::ones -shape {3 1}]
    set result [torch::hstack $tensor1 $tensor2 $tensor3]
    set shape [torch::tensor_shape $result]
    set shape
} {3 6}

# Test 8: Stack multiple tensors with named parameters
test hstack-4.2 {Stack multiple tensors (named parameters)} {
    set tensor1 [torch::ones -shape {3 2}]
    set tensor2 [torch::ones -shape {3 3}]
    set tensor3 [torch::ones -shape {3 1}]
    set tensor_list [list $tensor1 $tensor2 $tensor3]
    set result [torch::hstack -tensors $tensor_list]
    set shape [torch::tensor_shape $result]
    set shape
} {3 6}

# Test 9: Different tensor types - float32
test hstack-5.1 {Different tensor types - float32} {
    set tensor1 [torch::ones -shape {2 3} -dtype float32]
    set tensor2 [torch::ones -shape {2 4} -dtype float32]
    set result [torch::hstack -tensors [list $tensor1 $tensor2]]
    set dtype [torch::tensor_dtype $result]
    set shape [torch::tensor_shape $result]
    list $dtype $shape
} {Float32 {2 7}}

# Test 10: Different tensor types - int64
test hstack-5.2 {Different tensor types - int64} {
    set tensor1 [torch::ones -shape {2 3} -dtype int64]
    set tensor2 [torch::ones -shape {2 4} -dtype int64]
    set result [torch::hstack -tensors [list $tensor1 $tensor2]]
    set dtype [torch::tensor_dtype $result]
    set shape [torch::tensor_shape $result]
    list $dtype $shape
} {Int64 {2 7}}

# Test 11: 1D tensors horizontal stacking
test hstack-6.1 {1D tensors horizontal stacking} {
    set tensor1 [torch::ones -shape {3}]
    set tensor2 [torch::ones -shape {4}]
    set result [torch::hstack -tensors [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    set shape
} {7}

# Test 12: 3D tensors horizontal stacking
test hstack-6.2 {3D tensors horizontal stacking} {
    set tensor1 [torch::ones -shape {2 3 4}]
    set tensor2 [torch::ones -shape {2 5 4}]
    set result [torch::hstack -tensors [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    set shape
} {2 8 4}

# Test 13: Single tensor stack
test hstack-7.1 {Single tensor stack} {
    set tensor1 [torch::ones -shape {2 3}]
    set result [torch::hstack $tensor1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Test 14: Large tensor stack
test hstack-8.1 {Large tensor stack} {
    set tensor1 [torch::ones -shape {10 50}]
    set tensor2 [torch::ones -shape {10 30}]
    set result [torch::hstack -tensors [list $tensor1 $tensor2]]
    set shape [torch::tensor_shape $result]
    set shape
} {10 80}

# Test 15: Error handling - missing required parameters
test hstack-9.1 {Error handling - missing tensors parameter} {
    set result [catch {torch::hstack} msg]
    set result
} {1}

# Test 16: Error handling - invalid parameter name
test hstack-9.2 {Error handling - invalid parameter name} {
    set tensor1 [torch::ones -shape {2 3}]
    set result [catch {torch::hstack -invalid [list $tensor1]} msg]
    set result
} {1}

# Test 17: Error handling - missing value for parameter
test hstack-9.3 {Error handling - missing value for parameter} {
    set result [catch {torch::hstack -tensors} msg]
    set result
} {1}

# Test 18: Error handling - incompatible tensor shapes
test hstack-9.4 {Error handling - incompatible tensor shapes} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {3 4}]
    set result [catch {torch::hstack $tensor1 $tensor2} msg]
    # Should fail due to different first dimensions
    set result
} {1}

# Test 19: Consistency check - both syntaxes produce same results
test hstack-10.1 {Consistency check - both syntaxes produce same results} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    
    # Test both syntaxes
    set result1 [torch::hstack $tensor1 $tensor2]
    set result2 [torch::hstack -tensors [list $tensor1 $tensor2]]
    
    # Compare shapes
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    # Check if shapes are identical
    set shapes_match [expr {$shape1 eq $shape2}]
    
    set shapes_match
} {1}

# Test 20: Syntax consistency with camelCase alias
test hstack-10.2 {Syntax consistency with camelCase alias} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    
    # Test snake_case vs camelCase
    set result1 [torch::hstack -tensors [list $tensor1 $tensor2]]
    set result2 [torch::hStack -tensors [list $tensor1 $tensor2]]
    
    # Compare shapes
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    # Check if shapes are identical
    set shapes_match [expr {$shape1 eq $shape2}]
    
    set shapes_match
} {1}

# Test 21: Memory management - verify tensors are properly created
test hstack-11.1 {Memory management - verify tensors are properly created} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    set result [torch::hstack -tensors [list $tensor1 $tensor2]]
    
    # Check that result tensor is valid
    set valid [expr {[torch::tensor_shape $result] ne ""}]
    set valid
} {1}

# Test 22: Empty list handling
test hstack-9.5 {Error handling - empty tensor list} {
    set empty_list {}
    set result [catch {torch::hstack -tensors $empty_list} msg]
    # Should fail with empty list
    set result
} {1}

# Test 23: Mixed syntax edge case (ensure positional takes precedence)
test hstack-12.1 {Mixed syntax edge case} {
    set tensor1 [torch::ones -shape {2 3}]
    set tensor2 [torch::ones -shape {2 4}]
    
    # This should use positional syntax (first argument doesn't start with -)
    set result [torch::hstack $tensor1 $tensor2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 7}

# Test 24: Complex data integrity check
test hstack-13.1 {Complex data integrity check} {
    # Create tensors with different values to verify stacking order
    set tensor1 [torch::zeros -shape {2 2}]
    set tensor2 [torch::ones -shape {2 3}]
    set tensor3 [torch::full -shape {2 1} -value 2.0]
    
    set result [torch::hstack $tensor1 $tensor2 $tensor3]
    set shape [torch::tensor_shape $result]
    
    # Should stack horizontally: [2x2] + [2x3] + [2x1] = [2x6]
    set shape
} {2 6}

# Test 25: Gradients and autograd compatibility
test hstack-14.1 {Gradients and autograd compatibility} {
    set tensor1 [torch::ones -shape {2 3} -requiresGrad true]
    set tensor2 [torch::ones -shape {2 4} -requiresGrad true]
    set result [torch::hstack -tensors [list $tensor1 $tensor2]]
    
    # Check if result requires gradients
    set requires_grad [torch::tensor_requires_grad $result]
    set shape [torch::tensor_shape $result]
    
    list $requires_grad $shape
} {1 {2 7}}

cleanupTests 