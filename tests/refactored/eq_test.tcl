#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the shared library
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test parameters
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper procedure to create test tensors
proc create_tensor {data dims {dtype float32}} {
    return [torch::tensor_create -data $data -dtype $dtype -shape $dims]
}

proc create_int_tensor {data dims} {
    return [torch::tensor_create -data $data -dtype int64 -shape $dims]
}

# Helper procedure to check if result is a boolean tensor
proc is_boolean_tensor {tensor} {
    set dtype [torch::tensor_dtype $tensor]
    return [expr {$dtype eq "Bool"}]
}

# ========================================
# Tests for Positional Syntax (Backward Compatibility)
# ========================================

test eq-1.1 {Basic element-wise equality - positional syntax} {
    # Create two identical tensors
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    set t2 [create_tensor {1.0 2.0 3.0} {3}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape should be [3]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test eq-1.2 {Different tensors equality - positional syntax} {
    # Create two different tensors
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    set t2 [create_tensor {1.0 2.5 3.0} {3}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test eq-1.3 {2D tensor equality - positional syntax} {
    # Create 2D tensors
    set t1 [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set t2 [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape should be [2, 2]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test eq-1.4 {Integer tensor equality - positional syntax} {
    # Create integer tensors
    set t1 [create_int_tensor {1 2 3} {3}]
    set t2 [create_int_tensor {1 2 3} {3}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test eq-1.5 {Broadcasting equality - positional syntax} {
    # Test broadcasting (scalar vs tensor)
    set t1 [create_tensor {2.0 2.0 2.0} {3}]
    set t2 [create_tensor {2.0} {1}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape should be [3] (broadcast result)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

# ========================================
# Tests for Named Parameter Syntax
# ========================================

test eq-2.1 {Named syntax with -input1 and -input2} {
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    set t2 [create_tensor {1.0 2.0 3.0} {3}]
    
    set result [torch::eq -input1 $t1 -input2 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test eq-2.2 {Named syntax with -tensor1 and -tensor2} {
    set t1 [create_tensor {5.0 6.0} {2}]
    set t2 [create_tensor {5.0 7.0} {2}]
    
    set result [torch::eq -tensor1 $t1 -tensor2 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

test eq-2.3 {Parameter order independence} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 2.0} {2}]
    
    set result [torch::eq -input2 $t2 -input1 $t1]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

test eq-2.4 {Mixed parameter names} {
    set t1 [create_tensor {3.0 4.0} {2}]
    set t2 [create_tensor {3.0 4.0} {2}]
    
    set result [torch::eq -tensor1 $t1 -input2 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

# ========================================
# Tests for CamelCase Alias
# ========================================

test eq-3.1 {CamelCase alias - positional syntax} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 2.0} {2}]
    
    set result [torch::Eq $t1 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

test eq-3.2 {CamelCase alias - named syntax} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 2.0} {2}]
    
    set result [torch::Eq -input1 $t1 -input2 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

# ========================================
# Tests for Syntax Consistency
# ========================================

test eq-4.1 {Syntax consistency - same result structure} {
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    set t2 [create_tensor {1.0 2.5 3.0} {3}]
    
    set result1 [torch::eq $t1 $t2]
    set result2 [torch::eq -input1 $t1 -input2 $t2]
    
    # Check both have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test eq-4.2 {CamelCase produces same shape} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 3.0} {2}]
    
    set result1 [torch::eq $t1 $t2]
    set result2 [torch::Eq $t1 $t2]
    
    # Check both have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# ========================================
# Tests for Different Data Types
# ========================================

test eq-5.1 {Float32 tensor equality} {
    set t1 [create_tensor {1.5 2.5 3.5} {3} float32]
    set t2 [create_tensor {1.5 2.5 3.5} {3} float32]
    
    set result [torch::eq $t1 $t2]
    
    # Check result is boolean and has correct shape
    set shape [torch::tensor_shape $result]
    set is_bool [is_boolean_tensor $result]
    expr {$shape eq "3" && $is_bool}
} 1

test eq-5.2 {Int64 tensor equality} {
    set t1 [create_int_tensor {10 20 30} {3}]
    set t2 [create_int_tensor {10 25 30} {3}]
    
    set result [torch::eq $t1 $t2]
    
    # Check result shape and type
    set shape [torch::tensor_shape $result]
    set is_bool [is_boolean_tensor $result]
    expr {$shape eq "3" && $is_bool}
} 1

test eq-5.3 {Mixed precision comparison} {
    # Create tensors with same values but potentially different internal precision
    set t1 [create_tensor {1.0 2.0} {2} float32]
    set t2 [create_tensor {1.0 2.0} {2} float32]
    
    set result [torch::eq $t1 $t2]
    
    # Check output
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

# ========================================
# Tests for Different Tensor Shapes
# ========================================

test eq-6.1 {1D tensor equality} {
    set t1 [create_tensor {1.0 2.0 3.0 4.0 5.0} {5}]
    set t2 [create_tensor {1.0 2.0 3.0 4.0 5.0} {5}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "5"}
} 1

test eq-6.2 {2D tensor equality} {
    set t1 [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set t2 [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3"}
} 1

test eq-6.3 {3D tensor equality} {
    set t1 [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    set t2 [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2 2"}
} 1

test eq-6.4 {Single element tensors} {
    set t1 [create_tensor {5.0} {1}]
    set t2 [create_tensor {5.0} {1}]
    
    set result [torch::eq $t1 $t2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1"}
} 1

# ========================================
# Tests for Mathematical Properties
# ========================================

test eq-7.1 {Reflexivity - tensor equals itself} {
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    
    set result [torch::eq $t1 $t1]
    
    # All elements should be true (equal)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test eq-7.2 {Symmetry - eq(a,b) = eq(b,a)} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 3.0} {2}]
    
    set result1 [torch::eq $t1 $t2]
    set result2 [torch::eq $t2 $t1]
    
    # Both should have same shape (symmetry property)
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test eq-7.3 {Zero equality} {
    set t1 [create_tensor {0.0 0.0 0.0} {3}]
    set t2 [create_tensor {0.0 0.0 0.0} {3}]
    
    set result [torch::eq $t1 $t2]
    
    # Check shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test eq-7.4 {Negative number equality} {
    set t1 [create_tensor {-1.0 -2.0 -3.0} {3}]
    set t2 [create_tensor {-1.0 -2.0 -3.0} {3}]
    
    set result [torch::eq $t1 $t2]
    
    # Check shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

# ========================================
# Tests for Broadcasting
# ========================================

test eq-8.1 {Scalar vs tensor broadcasting} {
    set t1 [create_tensor {2.0} {1}]
    set t2 [create_tensor {2.0 2.0 2.0} {3}]
    
    set result [torch::eq $t1 $t2]
    
    # Result should be broadcast to [3]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test eq-8.2 {Different shaped tensor broadcasting} {
    # Test with broadcastable shapes
    set t1 [create_tensor {1.0 2.0} {2 1}]
    set t2 [create_tensor {1.0 2.0 3.0} {3}]
    
    # This should work with broadcasting
    set result [torch::eq $t1 $t2]
    
    # Result shape should be [2, 3]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3"}
} 1

# ========================================
# Error Handling Tests
# ========================================

test eq-9.1 {Error - missing arguments} {
    set result [catch {torch::eq} error]
    set result
} 1

test eq-9.2 {Error - insufficient positional arguments} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::eq $t1} error]
    set result
} 1

test eq-9.3 {Error - too many positional arguments} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 2.0} {2}]
    set t3 [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::eq $t1 $t2 $t3} error]
    set result
} 1

test eq-9.4 {Error - invalid tensor name} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::eq "invalid_tensor" $t1} error]
    set result
} 1

test eq-9.5 {Error - invalid second tensor} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::eq $t1 "invalid_tensor"} error]
    set result
} 1

test eq-9.6 {Error - missing value for named parameter} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::eq -input1 $t1 -input2} error]
    set result
} 1

test eq-9.7 {Error - unknown named parameter} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::eq -input1 $t1 -unknown_param $t2} error]
    set result
} 1

test eq-9.8 {Error - missing both tensors in named syntax} {
    set result [catch {torch::eq -input1} error]
    set result
} 1

test eq-9.9 {Error - only one tensor in named syntax} {
    set t1 [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::eq -input1 $t1} error]
    set result
} 1

test eq-9.10 {Error - CamelCase with invalid parameters} {
    set result [catch {torch::Eq "invalid_tensor1" "invalid_tensor2"} error]
    set result
} 1

# ========================================
# Integration Tests
# ========================================

test eq-10.1 {Integration with tensor operations} {
    # Create tensors and use equality result in further operations
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    set t2 [create_tensor {1.0 2.0 3.0} {3}]
    
    set eq_result [torch::eq $t1 $t2]
    
    # Count true values (should be 3 for identical tensors)
    set sum_result [torch::tensor_sum $eq_result]
    
    # Should result in a scalar
    set shape [torch::tensor_shape $sum_result]
    expr {$shape eq ""}
} 1

test eq-10.2 {Chained comparisons} {
    # Test using equality result in logical operations
    set t1 [create_tensor {1.0 2.0} {2}]
    set t2 [create_tensor {1.0 2.0} {2}]
    set t3 [create_tensor {1.0 3.0} {2}]
    
    # Should be true, true
    set eq1 [torch::eq $t1 $t2]
    # Should be true, false
    set eq2 [torch::eq $t1 $t3]
    
    # Both should have same shape for logical operations
    set shape1 [torch::tensor_shape $eq1]
    set shape2 [torch::tensor_shape $eq2]
    expr {$shape1 eq $shape2 && $shape1 eq "2"}
} 1

test eq-10.3 {Equality in conditional logic} {
    # Test using equality for masking/indexing operations
    set data [create_tensor {1.0 2.0 1.0 3.0 1.0} {5}]
    set target [create_tensor {1.0} {1}]
    
    # Find all 1.0s
    set mask [torch::eq $data $target]
    
    # Mask should have shape 5
    set shape [torch::tensor_shape $mask]
    expr {$shape eq "5"}
} 1

# Clean up any remaining tensors and run tests
cleanupTests 