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

# Helper procedure to check tensor values approximately
proc tensor_approx_equal {t1 t2 {tolerance 1e-6}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# Helper to extract tensor values for checking
proc tensor_values {tensor} {
    # For now, we'll just use shape checking and mathematical properties
    # rather than direct value extraction since tensor_data doesn't exist
    return [torch::tensor_shape $tensor]
}

# ========================================
# Tests for Positional Syntax (Backward Compatibility)
# ========================================

test einsum-1.1 {Basic matrix multiplication - positional syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set b [create_tensor {5.0 6.0 7.0 8.0} {2 2}]
    set result [torch::einsum "ij,jk->ik" $a $b]
    
    # Check shape and verify it's a valid tensor
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test einsum-1.2 {Trace computation - positional syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::einsum "ii->" $a]
    
    # Expected: 5.0 (1 + 4), check it's a scalar
    set shape [torch::tensor_shape $result]
    set trace_val [torch::tensor_item $result]
    expr {$shape eq "" && abs($trace_val - 5.0) < 1e-6}
} 1

test einsum-1.3 {Transpose operation - positional syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::einsum "ij->ji" $a]
    
    # Should transpose from 2x3 to 3x2
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

test einsum-1.4 {Element-wise multiplication - positional syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set b [create_tensor {5.0 6.0 7.0 8.0} {2 2}]
    set result [torch::einsum "ij,ij->ij" $a $b]
    
    # Should preserve shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test einsum-1.5 {Batch matrix multiplication - positional syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    set b [create_tensor {1.0 0.0 0.0 1.0 2.0 0.0 0.0 2.0} {2 2 2}]
    set result [torch::einsum "bij,bjk->bik" $a $b]
    
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2 2"}
} 1

# ========================================
# Tests for Named Parameter Syntax
# ========================================

test einsum-2.1 {Matrix multiplication - named syntax with list} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set b [create_tensor {5.0 6.0 7.0 8.0} {2 2}]
    set result [torch::einsum -equation "ij,jk->ik" -tensors [list $a $b]]
    
    # Check shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test einsum-2.2 {Matrix multiplication - named syntax single tensor} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::einsum -equation "ii->" -tensors $a]
    
    # Expected: 5.0 (1 + 4)
    set trace_val [torch::tensor_item $result]
    expr {abs($trace_val - 5.0) < 1e-6}
} 1

test einsum-2.3 {Sum over axis - named syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::einsum -equation "ij->i" -tensors [list $a]]
    
    # Should result in shape [2]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

test einsum-2.4 {Three tensor contraction - named syntax} {
    set a [create_tensor {1.0 2.0} {2}]
    set b [create_tensor {3.0 4.0} {2}]
    set c [create_tensor {5.0 6.0} {2}]
    set result [torch::einsum -equation "i,i,i->" -tensors [list $a $b $c]]
    
    # Expected: 1*3*5 + 2*4*6 = 15 + 48 = 63
    set val [torch::tensor_item $result]
    expr {abs($val - 63.0) < 1e-6}
} 1

# ========================================
# Tests for CamelCase Alias
# ========================================

test einsum-3.1 {CamelCase alias - positional syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set b [create_tensor {5.0 6.0 7.0 8.0} {2 2}]
    set result [torch::Einsum "ij,jk->ik" $a $b]
    
    # Check shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test einsum-3.2 {CamelCase alias - named syntax} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::Einsum -equation "ii->" -tensors $a]
    
    # Expected: 5.0 (1 + 4)
    set trace_val [torch::tensor_item $result]
    expr {abs($trace_val - 5.0) < 1e-6}
} 1

# ========================================
# Tests for Syntax Consistency
# ========================================

test einsum-4.1 {Syntax consistency - same result from both syntaxes} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set b [create_tensor {5.0 6.0 7.0 8.0} {2 2}]
    
    set result1 [torch::einsum "ij,jk->ik" $a $b]
    set result2 [torch::einsum -equation "ij,jk->ik" -tensors [list $a $b]]
    
    # Check results are approximately equal
    tensor_approx_equal $result1 $result2 1e-10
} 1

test einsum-4.2 {CamelCase produces same result} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    
    set result1 [torch::einsum "ii->" $a]
    set result2 [torch::Einsum "ii->" $a]
    
    set val1 [torch::tensor_item $result1]
    set val2 [torch::tensor_item $result2]
    expr {abs($val1 - $val2) < 1e-10}
} 1

# ========================================
# Tests for Advanced Operations
# ========================================

test einsum-5.1 {Outer product} {
    set a [create_tensor {1.0 2.0} {2}]
    set b [create_tensor {3.0 4.0} {2}]
    set result [torch::einsum "i,j->ij" $a $b]
    
    # Should result in 2x2 matrix
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test einsum-5.2 {Diagonal extraction} {
    set a [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} {3 3}]
    set result [torch::einsum "ii->i" $a]
    
    # Should extract diagonal to vector of length 3
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

test einsum-5.3 {Sum all elements} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::einsum "ij->" $a]
    
    # Should sum to 10.0
    set val [torch::tensor_item $result]
    expr {abs($val - 10.0) < 1e-6}
} 1

# ========================================
# Error Handling Tests
# ========================================

test einsum-6.1 {Error - missing arguments} {
    set result [catch {torch::einsum} error]
    set result
} 1

test einsum-6.2 {Error - missing equation in positional syntax} {
    set result [catch {torch::einsum "ij"} error]
    set result
} 1

test einsum-6.3 {Error - invalid tensor in positional syntax} {
    set result [catch {torch::einsum "ij->" "invalid_tensor"} error]
    set result
} 1

test einsum-6.4 {Error - missing value for named parameter} {
    set result [catch {torch::einsum -equation} error]
    set result
} 1

test einsum-6.5 {Error - unknown parameter} {
    set a [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::einsum -invalid_param $a} error]
    set result
} 1

test einsum-6.6 {Error - invalid tensor in named syntax} {
    set result [catch {torch::einsum -equation "i->" -tensors "invalid_tensor"} error]
    set result
} 1

test einsum-6.7 {Error - empty equation} {
    set a [create_tensor {1.0 2.0} {2}]
    set result [catch {torch::einsum -equation "" -tensors $a} error]
    set result
} 1

test einsum-6.8 {Error - malformed equation} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [catch {torch::einsum "invalid_equation" $a} error]
    set result
} 1

test einsum-6.9 {Error - dimension mismatch} {
    set a [create_tensor {1.0 2.0} {2}]
    set b [create_tensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::einsum "i,i->i" $a $b} error]
    set result
} 1

test einsum-6.10 {Error - CamelCase with invalid tensor} {
    set result [catch {torch::Einsum "i->" "invalid_tensor"} error]
    set result
} 1

# ========================================
# Integration Tests
# ========================================

test einsum-7.1 {Integration with other tensor operations} {
    set a [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set b [create_tensor {2.0 0.0 0.0 2.0} {2 2}]
    
    # Multiply matrices then compute trace
    set matmul [torch::einsum "ij,jk->ik" $a $b]
    set trace [torch::einsum "ii->" $matmul]
    
    # Expected: [[2, 4], [6, 8]] -> trace = 10
    set trace_val [torch::tensor_item $trace]
    expr {abs($trace_val - 10.0) < 1e-6}
} 1

# Clean up any remaining tensors and run tests
cleanupTests 