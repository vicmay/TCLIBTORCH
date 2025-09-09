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

# Helper procedures
proc tensor_equal {t1 t2 {tolerance 1e-6}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

proc create_test_tensor {values shape} {
    set tensor [torch::tensor_create -data $values -dtype float32 -device cpu -requiresGrad false]
    if {[llength $shape] > 1} {
        return [torch::tensor_reshape $tensor $shape]
    }
    return $tensor
}

# ===== Positional Syntax Tests =====

test logsoftmax-1.1 {Basic positional syntax - 1D tensor} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [torch::logsoftmax $input]
    
    # Verify result is valid and has correct shape
    set shape [torch::tensor_shape $result]
    expr {$shape == "3"}
} {1}

test logsoftmax-1.2 {Positional syntax with explicit dimension} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::logsoftmax $input 1]
    
    # Verify result shape matches input
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} {1}

test logsoftmax-1.3 {Positional syntax with default dimension} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::logsoftmax $input]
    
    # Should use last dimension (-1) by default
    set shape [torch::tensor_shape $result]
    expr {$shape == "2 2"}
} {1}

# ===== Named Parameter Syntax Tests =====

test logsoftmax-2.1 {Named syntax with -input parameter} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [torch::logsoftmax -input $input]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "3"}
} {1}

test logsoftmax-2.2 {Named syntax with -tensor parameter} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [torch::logsoftmax -tensor $input]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "3"}
} {1}

test logsoftmax-2.3 {Named syntax with -dim parameter} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::logsoftmax -input $input -dim 0]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "2 2"}
} {1}

test logsoftmax-2.4 {Named syntax with -dimension parameter} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::logsoftmax -input $input -dimension 1]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "2 2"}
} {1}

test logsoftmax-2.5 {Named syntax with all parameters} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    set result [torch::logsoftmax -input $input -dim 1]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "2 3"}
} {1}

# ===== camelCase Alias Tests =====

test logsoftmax-3.1 {camelCase alias - basic usage} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [torch::logSoftmax $input]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "3"}
} {1}

test logsoftmax-3.2 {camelCase alias with named parameters} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result [torch::logSoftmax -input $input -dim 0]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "2 2"}
} {1}

# ===== Consistency Tests =====

test logsoftmax-4.1 {Consistency between positional and named syntax} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result1 [torch::logsoftmax $input 1]
    set result2 [torch::logsoftmax -input $input -dim 1]
    
    tensor_equal $result1 $result2
} {1}

test logsoftmax-4.2 {Consistency between snake_case and camelCase} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result1 [torch::logsoftmax $input]
    set result2 [torch::logSoftmax $input]
    
    tensor_equal $result1 $result2
} {1}

test logsoftmax-4.3 {Consistency - named syntax with different parameter names} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set result1 [torch::logsoftmax -input $input -dim 0]
    set result2 [torch::logsoftmax -tensor $input -dimension 0]
    
    tensor_equal $result1 $result2
} {1}

# ===== Mathematical Properties Tests =====

test logsoftmax-5.1 {LogSoftmax mathematical properties - 1D case} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set logsoftmax_result [torch::logsoftmax $input]
    
    # Convert to softmax and check sum
    set softmax_result [torch::tensor_exp $logsoftmax_result]
    set sum_tensor [torch::tensor_sum $softmax_result]
    set sum_val [torch::tensor_item $sum_tensor]
    
    # Sum should be approximately 1.0
    expr {abs($sum_val - 1.0) < 1e-5}
} {1}

test logsoftmax-5.2 {LogSoftmax vs manual computation} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set logsoftmax_result [torch::logsoftmax $input]
    
    # Manual computation: log(softmax(x)) = x - log(sum(exp(x)))
    set exp_input [torch::tensor_exp $input]
    set sum_exp [torch::tensor_sum $exp_input]
    set log_sum_exp [torch::tensor_log $sum_exp]
    set manual_result [torch::tensor_sub $input $log_sum_exp]
    
    tensor_equal $logsoftmax_result $manual_result
} {1}

test logsoftmax-5.3 {LogSoftmax with different dimensions} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {2 3}]
    
    # Apply along dimension 0
    set result_dim0 [torch::logsoftmax $input 0]
    set shape0 [torch::tensor_shape $result_dim0]
    
    # Apply along dimension 1  
    set result_dim1 [torch::logsoftmax $input 1]
    set shape1 [torch::tensor_shape $result_dim1]
    
    # Both should preserve input shape
    expr {$shape0 == "2 3" && $shape1 == "2 3"}
} {1}

test logsoftmax-5.4 {LogSoftmax numerical stability} {
    # Test with large values that could cause overflow
    set input [create_test_tensor {100.0 101.0 102.0} {3}]
    set result [torch::logsoftmax $input]
    
    # Should not produce inf or nan
    set max_val [torch::tensor_item [torch::tensor_max $result]]
    set min_val [torch::tensor_item [torch::tensor_min $result]]
    
    expr {!([string equal $max_val "inf"] || [string equal $max_val "nan"] || 
           [string equal $min_val "inf"] || [string equal $min_val "nan"])}
} {1}

# ===== Edge Cases Tests =====

test logsoftmax-6.1 {Single element tensor} {
    set input [create_test_tensor {5.0} {1}]
    set result [torch::logsoftmax $input]
    
    # LogSoftmax of single element should be 0
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 1e-6}
} {1}

test logsoftmax-6.2 {Zero tensor} {
    set input [create_test_tensor {0.0 0.0 0.0} {3}]
    set result [torch::logsoftmax $input]
    
    # All elements should be equal (log(1/3))
    # Test by checking that all differences are close to zero
    set expected_val [expr {log(1.0/3.0)}]
    set expected_tensor [torch::full {3} $expected_val]
    tensor_equal $result $expected_tensor
} {1}

test logsoftmax-6.3 {Negative values} {
    set input [create_test_tensor {-1.0 -2.0 -3.0} {3}]
    set result [torch::logsoftmax $input]
    
    # Should work with negative inputs
    set shape [torch::tensor_shape $result]
    expr {$shape == "3"}
} {1}

# ===== Error Handling Tests =====

test logsoftmax-7.1 {Error - missing input parameter} {
    set result [catch {torch::logsoftmax} error]
    expr {$result == 1}
} {1}

test logsoftmax-7.2 {Error - invalid tensor name} {
    set result [catch {torch::logsoftmax invalid_tensor} error]
    expr {$result == 1}
} {1}

test logsoftmax-7.3 {Error - invalid dimension type} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::logsoftmax $input "invalid"} error]
    expr {$result == 1}
} {1}

test logsoftmax-7.4 {Error - missing value for named parameter} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::logsoftmax -input} error]
    expr {$result == 1}
} {1}

test logsoftmax-7.5 {Error - unknown parameter} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::logsoftmax -invalid $input} error]
    expr {$result == 1}
} {1}

test logsoftmax-7.6 {Error - invalid dimension value in named syntax} {
    set input [create_test_tensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::logsoftmax -input $input -dim "invalid"} error]
    expr {$result == 1}
} {1}

# ===== Data Type Tests =====

test logsoftmax-8.1 {Different data types - float64} {
    set input [torch::tensor_create {1.0 2.0 3.0} float64]
    set result [torch::logsoftmax $input]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "3"}
} {1}

test logsoftmax-8.2 {Error with integer tensors} {
    # LogSoftmax doesn't support integer dtypes in PyTorch
    set input [torch::tensor_create {1 2 3} int32]
    set result [catch {torch::logsoftmax $input} error]
    expr {$result == 1}
} {1}

# ===== Multi-dimensional Tests =====

test logsoftmax-9.1 {3D tensor with different dimensions} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set input [torch::tensor_reshape $input {2 2 2}]
    
    # Test dimension 0
    set result0 [torch::logsoftmax $input 0]
    set shape0 [torch::tensor_shape $result0]
    
    # Test dimension 1
    set result1 [torch::logsoftmax $input 1]
    set shape1 [torch::tensor_shape $result1]
    
    # Test dimension 2
    set result2 [torch::logsoftmax $input 2]
    set shape2 [torch::tensor_shape $result2]
    
    # All should preserve input shape
    expr {$shape0 == "2 2 2" && $shape1 == "2 2 2" && $shape2 == "2 2 2"}
} {1}

test logsoftmax-9.2 {Negative dimension indexing} {
    set input [create_test_tensor {1.0 2.0 3.0 4.0} {2 2}]
    
    # -1 should be equivalent to last dimension (1 in this case)
    set result_neg1 [torch::logsoftmax $input -1]
    set result_pos1 [torch::logsoftmax $input 1]
    
    tensor_equal $result_neg1 $result_pos1
} {1}

# ===== Performance/Large Tensor Tests =====

test logsoftmax-10.1 {Large tensor handling} {
    # Create larger tensor (100 elements)
    set values {}
    for {set i 0} {$i < 100} {incr i} {
        lappend values [expr {$i * 0.1}]
    }
    set input [create_test_tensor $values {10 10}]
    set result [torch::logsoftmax $input 1]
    
    set shape [torch::tensor_shape $result]
    expr {$shape == "10 10"}
} {1}

cleanupTests 