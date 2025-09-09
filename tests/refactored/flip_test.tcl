#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

proc createTestTensor {data shape} {
    return [torch::tensor_create -data $data -shape $shape -dtype float32]
}

# =============================================================================
# DUAL SYNTAX TESTS - POSITIONAL SYNTAX (BACKWARD COMPATIBLE)
# =============================================================================

test flip-1.1 {Basic positional syntax - flip single dimension} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result [torch::flip $tensor {0}]
    expr {$result ne ""}
} 1

test flip-1.2 {Positional syntax - flip 2D tensor along dimension 0} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::flip $tensor {0}]
    expr {$result ne ""}
} 1

test flip-1.3 {Positional syntax - flip 2D tensor along dimension 1} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::flip $tensor {1}]
    expr {$result ne ""}
} 1

test flip-1.4 {Positional syntax - flip multiple dimensions} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    set result [torch::flip $tensor {0 1}]
    expr {$result ne ""}
} 1

test flip-1.5 {Positional syntax - flip all dimensions} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    set result [torch::flip $tensor {0 1 2}]
    expr {$result ne ""}
} 1

# =============================================================================
# DUAL SYNTAX TESTS - NAMED PARAMETER SYNTAX
# =============================================================================

test flip-2.1 {Named syntax with -input and -dims} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result [torch::flip -input $tensor -dims {0}]
    expr {$result ne ""}
} 1

test flip-2.2 {Named syntax with -tensor and -dimensions alternative} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result [torch::flip -tensor $tensor -dimensions {0}]
    expr {$result ne ""}
} 1

test flip-2.3 {Named syntax - flip 2D tensor along dimension 0} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::flip -input $tensor -dims {0}]
    expr {$result ne ""}
} 1

test flip-2.4 {Named syntax - flip 2D tensor along dimension 1} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::flip -input $tensor -dims {1}]
    expr {$result ne ""}
} 1

test flip-2.5 {Named syntax - flip multiple dimensions} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    set result [torch::flip -input $tensor -dims {0 1}]
    expr {$result ne ""}
} 1

test flip-2.6 {Named syntax - parameter order independence} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result [torch::flip -dims {0} -input $tensor]
    expr {$result ne ""}
} 1

test flip-2.7 {Named syntax - mixed parameter naming styles} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::flip -tensor $tensor -dims {1}]
    expr {$result ne ""}
} 1

# =============================================================================
# CAMELCASE ALIAS TESTS
# =============================================================================

test flip-3.1 {CamelCase alias - positional syntax} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result [torch::Flip $tensor {0}]
    expr {$result ne ""}
} 1

test flip-3.2 {CamelCase alias - named syntax} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result [torch::Flip -input $tensor -dims {0}]
    expr {$result ne ""}
} 1

# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

test flip-4.1 {Syntax consistency - both should produce same result shape} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result1 [torch::flip $tensor {0}]
    set result2 [torch::flip -input $tensor -dims {0}]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2 && $shape1 eq "4"}
} 1

test flip-4.2 {camelCase alias consistency} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result1 [torch::flip $tensor {0}]
    set result2 [torch::Flip $tensor {0}]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} 1

# =============================================================================
# FUNCTIONAL TESTS
# =============================================================================

test flip-5.1 {Flip 1D tensor functionality} {
    set tensor [createTestTensor {1.0 2.0 3.0} {3}]
    set result [torch::flip $tensor {0}]
    
    # Verify result has correct shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test flip-5.2 {Flip 2D tensor along rows} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::flip $tensor {0}]
    
    # Verify result has correct shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test flip-5.3 {Flip 2D tensor along columns} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::flip $tensor {1}]
    
    # Verify result has correct shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test flip-5.4 {Flip 3D tensor along multiple dimensions} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    set result [torch::flip $tensor {0 2}]
    
    # Verify result has correct shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2 2"}
} 1

test flip-5.5 {Flip with dimension index (positive only)} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::flip $tensor {1}]
    
    # Verify result has correct shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

# =============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# =============================================================================

test flip-6.1 {Double flip returns original tensor shape} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set flipped [torch::flip $tensor {0}]
    set double_flipped [torch::flip $flipped {0}]
    
    # Should have same shape as original
    set orig_shape [torch::tensor_shape $tensor]
    set final_shape [torch::tensor_shape $double_flipped]
    expr {$orig_shape eq $final_shape}
} 1

test flip-6.2 {Flip preserves tensor dtype} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {4}]
    set result [torch::flip $tensor {0}]
    
    set orig_dtype [torch::tensor_dtype $tensor]
    set result_dtype [torch::tensor_dtype $result]
    expr {$orig_dtype eq $result_dtype}
} 1

test flip-6.3 {Flip different dimensions independently} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    
    set flip_dim0 [torch::flip $tensor {0}]
    set flip_dim1 [torch::flip $tensor {1}]
    
    # Both should preserve original shape
    set orig_shape [torch::tensor_shape $tensor]
    set shape0 [torch::tensor_shape $flip_dim0]
    set shape1 [torch::tensor_shape $flip_dim1]
    
    expr {$orig_shape eq $shape0 && $orig_shape eq $shape1}
} 1

# =============================================================================
# DIMENSION VALIDATION TESTS
# =============================================================================

test flip-7.1 {Various dimension configurations - single dimension} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    
    set result0 [torch::flip $tensor {0}]
    set result1 [torch::flip $tensor {1}]
    set result2 [torch::flip $tensor {2}]
    
    expr {$result0 ne "" && $result1 ne "" && $result2 ne ""}
} 1

test flip-7.2 {Multiple dimension combinations} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    
    set result01 [torch::flip $tensor {0 1}]
    set result02 [torch::flip $tensor {0 2}]
    set result12 [torch::flip $tensor {1 2}]
    
    expr {$result01 ne "" && $result02 ne "" && $result12 ne ""}
} 1

test flip-7.3 {All dimensions flipped} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    set result [torch::flip $tensor {0 1 2}]
    
    # Should preserve shape
    set orig_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    expr {$orig_shape eq $result_shape}
} 1

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

test flip-8.1 {Error - missing arguments} {
    set result [catch {torch::flip} error]
    set result
} 1

test flip-8.2 {Error - insufficient arguments for positional syntax} {
    set tensor [createTestTensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::flip $tensor} error]
    set result
} 1

test flip-8.3 {Error - invalid tensor name} {
    set result [catch {torch::flip "invalid_tensor" {0}} error]
    set result
} 1

test flip-8.4 {Error - missing value for named parameter} {
    set tensor [createTestTensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::flip -input $tensor -dims} error]
    set result
} 1

test flip-8.5 {Error - unknown named parameter} {
    set tensor [createTestTensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::flip -input $tensor -invalid {0}} error]
    set result
} 1

test flip-8.6 {Error - invalid dimension value} {
    set tensor [createTestTensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::flip $tensor {"invalid"}} error]
    set result
} 1

test flip-8.7 {Error - empty dimensions list} {
    set tensor [createTestTensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::flip -input $tensor -dims {}} error]
    set result
} 1

# =============================================================================
# INTEGRATION TESTS  
# =============================================================================

test flip-9.1 {Integration with other tensor operations} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0} {2 2}]
    
    # Chain operations: create -> flip -> get shape
    set flipped [torch::flip $tensor {0}]
    set shape [torch::tensor_shape $flipped]
    
    expr {$shape eq "2 2"}
} 1

test flip-9.2 {Multiple flip operations} {
    set tensor [createTestTensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    
    # Multiple flips with different syntaxes
    set flip1 [torch::flip $tensor {0}]
    set flip2 [torch::flip -input $flip1 -dims {1}]
    set flip3 [torch::Flip $flip2 {0}]
    
    # All should have same shape as original
    set orig_shape [torch::tensor_shape $tensor]
    set final_shape [torch::tensor_shape $flip3]
    expr {$orig_shape eq $final_shape}
} 1

test flip-9.3 {Flip with different tensor types} {
    # Test with integer tensor
    set int_tensor [torch::tensor_create -data {1 2 3 4} -shape {2 2} -dtype int32]
    set int_result [torch::flip $int_tensor {0}]
    
    # Test with float tensor  
    set float_tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set float_result [torch::flip $float_tensor {1}]
    
    expr {$int_result ne "" && $float_result ne ""}
} 1

# Clean up any remaining tensors and run tests
cleanupTests 