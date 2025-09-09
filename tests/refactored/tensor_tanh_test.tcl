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

# Test cases for positional syntax (backward compatibility)
test tensor_tanh-1.1 {Basic positional syntax - zero value} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    expr {abs($value) < 0.001}
} {1}

test tensor_tanh-1.2 {Positional syntax - positive value} {
    set t1 [torch::full {1} 1.0]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    # tanh(1) ≈ 0.7616
    expr {abs($value - 0.7616) < 0.001}
} {1}

test tensor_tanh-1.3 {Positional syntax - negative value} {
    set t1 [torch::full {1} -1.0]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    # tanh(-1) ≈ -0.7616
    expr {abs($value + 0.7616) < 0.001}
} {1}

test tensor_tanh-1.4 {Positional syntax - large positive} {
    set t1 [torch::full {1} 10.0]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    # tanh(large) = 1.0 due to floating-point precision
    expr {$value == 1.0}
} {1}

test tensor_tanh-1.5 {Positional syntax - large negative} {
    set t1 [torch::full {1} -10.0]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    # tanh(-large) = -1.0 due to floating-point precision
    expr {$value == -1.0}
} {1}

test tensor_tanh-1.6 {Positional syntax with multi-element tensor} {
    set t1 [torch::full {2 2} 1.0]
    set result [torch::tensor_tanh $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

# Test cases for named parameter syntax
test tensor_tanh-2.1 {Named parameter syntax - zero value} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_tanh -input $t1]
    set value [torch::tensor_item $result]
    expr {abs($value) < 0.001}
} {1}

test tensor_tanh-2.2 {Named parameter syntax - positive value} {
    set t1 [torch::full {1} 2.0]
    set result [torch::tensor_tanh -input $t1]
    set value [torch::tensor_item $result]
    # tanh(2) ≈ 0.9640
    expr {abs($value - 0.9640) < 0.001}
} {1}

test tensor_tanh-2.3 {Named parameter syntax preserves tensor properties} {
    set t1 [torch::full {2 3} 0.5]
    set result [torch::tensor_tanh -input $t1]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "2 3" && [string match "*Float*" $dtype]}
} {1}

# Test cases for camelCase alias
test tensor_tanh-3.1 {CamelCase alias syntax - positive} {
    set t1 [torch::full {1} 0.5]
    set result [torch::tensorTanh $t1]
    set value [torch::tensor_item $result]
    # tanh(0.5) ≈ 0.4621
    expr {abs($value - 0.4621) < 0.001}
} {1}

test tensor_tanh-3.2 {CamelCase with named parameters - negative} {
    set t1 [torch::full {1} -0.5]
    set result [torch::tensorTanh -input $t1]
    set value [torch::tensor_item $result]
    # tanh(-0.5) ≈ -0.4621
    expr {abs($value + 0.4621) < 0.001}
} {1}

# Error handling tests
test tensor_tanh-4.1 {Error: invalid tensor name (positional)} {
    catch {torch::tensor_tanh invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_tanh-4.2 {Error: invalid tensor name (named)} {
    catch {torch::tensor_tanh -input invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_tanh-4.3 {Error: missing required parameter} {
    catch {torch::tensor_tanh} result
    expr {[string match "*Usage*" $result] || [string match "*Required parameter*" $result]}
} {1}

test tensor_tanh-4.4 {Error: unknown parameter} {
    set t1 [torch::ones {1}]
    catch {torch::tensor_tanh -invalid_param $t1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test tensor_tanh-4.5 {Error: missing value for named parameter} {
    catch {torch::tensor_tanh -input} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

# Mathematical correctness tests
test tensor_tanh-5.1 {Mathematical properties: range [-1,1]} {
    # tanh is bounded between -1 and 1 (inclusive due to floating-point)
    set t1 [torch::full {1} 100.0]
    set t2 [torch::full {1} -100.0]
    set result_large [torch::tensor_tanh $t1]
    set result_small [torch::tensor_tanh $t2]
    set value_large [torch::tensor_item $result_large]
    set value_small [torch::tensor_item $result_small]
    
    expr {$value_large <= 1.0 && $value_large >= 0.0 && $value_small >= -1.0 && $value_small <= 0.0}
} {1}

test tensor_tanh-5.2 {Mathematical properties: odd function} {
    # tanh(-x) = -tanh(x)
    set t1 [torch::full {1} 1.5]
    set t2 [torch::full {1} -1.5]
    set result_pos [torch::tensor_tanh $t1]
    set result_neg [torch::tensor_tanh $t2]
    set value_pos [torch::tensor_item $result_pos]
    set value_neg [torch::tensor_item $result_neg]
    
    expr {abs($value_pos + $value_neg) < 0.0001}
} {1}

test tensor_tanh-5.3 {Mathematical properties: tanh(0) = 0} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    expr {abs($value) < 0.0001}
} {1}

# Data type consistency tests
test tensor_tanh-6.1 {Different data types: float64} {
    set t1 [torch::full {1} 1.0 float64]
    set result [torch::tensor_tanh $t1]
    set dtype [torch::tensor_dtype $result]
    set value [torch::tensor_item $result]
    expr {[string match "*Float64*" $dtype] && abs($value - 0.7616) < 0.001}
} {1}

# Integration tests with other commands
test tensor_tanh-7.1 {Chain operations: tanh after multiplication} {
    set t1 [torch::full {1} 0.5]
    set t2 [torch::full {1} 2.0]
    # 0.5 * 2 = 1.0
    set prod [torch::tensor_mul $t1 $t2]
    set tanh_result [torch::tensor_tanh $prod]
    set value [torch::tensor_item $tanh_result]
    # tanh(1.0) ≈ 0.7616
    expr {abs($value - 0.7616) < 0.001}
} {1}

test tensor_tanh-7.2 {Neural network pattern: linear + tanh} {
    # Simulate: input * weight + bias, then tanh
    set input [torch::full {1} 1.0]
    set weight [torch::full {1} 0.5]
    set bias [torch::full {1} 0.0]
    
    # 1.0 * 0.5 = 0.5
    set linear_output [torch::tensor_mul $input $weight]
    # 0.5 + 0.0 = 0.5
    set with_bias [torch::tensor_add $linear_output $bias]
    set activated [torch::tensor_tanh $with_bias]
    set value [torch::tensor_item $activated]
    # tanh(0.5) ≈ 0.4621
    expr {abs($value - 0.4621) < 0.001}
} {1}

# Syntax consistency tests
test tensor_tanh-8.1 {Both syntaxes produce same result} {
    set t1 [torch::full {1} 0.75]
    set result1 [torch::tensor_tanh $t1]
    set result2 [torch::tensor_tanh -input $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test tensor_tanh-8.2 {CamelCase alias produces same result} {
    set t1 [torch::full {1} -0.75]
    set result1 [torch::tensor_tanh $t1]
    set result2 [torch::tensorTanh $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {abs($value1 - $value2) < 0.000001}
} {1}

# Edge cases and special values
test tensor_tanh-9.1 {Edge case: very small positive value} {
    set t1 [torch::full {1} 0.0001]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    # tanh(small) ≈ small for small values
    expr {abs($value - 0.0001) < 0.0001}
} {1}

test tensor_tanh-9.2 {Edge case: very small negative value} {
    set t1 [torch::full {1} -0.0001]
    set result [torch::tensor_tanh $t1]
    set value [torch::tensor_item $result]
    # tanh(-small) ≈ -small for small values
    expr {abs($value + 0.0001) < 0.0001}
} {1}

test tensor_tanh-9.3 {Asymptotic behavior} {
    # Test that tanh approaches ±1 for large values
    set pos_large [torch::full {1} 100.0]
    set neg_large [torch::full {1} -100.0]
    
    set pos_result [torch::tensor_tanh $pos_large]
    set neg_result [torch::tensor_tanh $neg_large]
    
    set pos_val [torch::tensor_item $pos_result]
    set neg_val [torch::tensor_item $neg_result]
    
    # Due to floating-point precision, these actually reach exactly ±1.0
    expr {$pos_val == 1.0 && $neg_val == -1.0}
} {1}

# Gradient behavior verification (mathematical property)
test tensor_tanh-10.1 {Derivative relationship: 1 - tanh²(x)} {
    # tanh'(x) = 1 - tanh²(x)
    set t1 [torch::full {1} 0.0]
    set tanh_result [torch::tensor_tanh $t1]
    set tanh_val [torch::tensor_item $tanh_result]
    
    # At x=0, derivative = 1 - tanh²(0) = 1 - 0² = 1
    set expected_derivative [expr {1.0 - ($tanh_val * $tanh_val)}]
    
    expr {abs($expected_derivative - 1.0) < 0.001}
} {1}

cleanupTests 