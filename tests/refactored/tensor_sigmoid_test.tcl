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
test tensor_sigmoid-1.1 {Basic positional syntax - sigmoid(0) = 0.5} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.5) < 0.001}
} {1}

test tensor_sigmoid-1.2 {Positional syntax - large positive value} {
    set t1 [torch::full {1} 5.0]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    # sigmoid(5) ≈ 0.9933
    expr {$value > 0.99 && $value < 1.0}
} {1}

test tensor_sigmoid-1.3 {Positional syntax - large negative value} {
    set t1 [torch::full {1} -5.0]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    # sigmoid(-5) ≈ 0.0067
    expr {$value > 0.0 && $value < 0.01}
} {1}

test tensor_sigmoid-1.4 {Positional syntax - positive moderate value} {
    set t1 [torch::full {1} 2.0]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    # sigmoid(2) ≈ 0.8808
    expr {abs($value - 0.8808) < 0.01}
} {1}

test tensor_sigmoid-1.5 {Positional syntax - negative moderate value} {
    set t1 [torch::full {1} -2.0]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    # sigmoid(-2) ≈ 0.1192
    expr {abs($value - 0.1192) < 0.01}
} {1}

test tensor_sigmoid-1.6 {Positional syntax with multi-element tensor} {
    set t1 [torch::full {2 2} 1.0]
    set result [torch::tensor_sigmoid $t1]
    set shape [torch::tensor_shape $result]
    # Check that shape is preserved
    expr {$shape eq "2 2"}
} {1}

# Test cases for named parameter syntax
test tensor_sigmoid-2.1 {Named parameter syntax - sigmoid(0)} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_sigmoid -input $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.5) < 0.001}
} {1}

test tensor_sigmoid-2.2 {Named parameter syntax - positive value} {
    set t1 [torch::full {1} 1.0]
    set result [torch::tensor_sigmoid -input $t1]
    set value [torch::tensor_item $result]
    # sigmoid(1) ≈ 0.7311
    expr {abs($value - 0.7311) < 0.01}
} {1}

test tensor_sigmoid-2.3 {Named parameter syntax preserves tensor properties} {
    set t1 [torch::full {2 3} 0.5]
    set result [torch::tensor_sigmoid -input $t1]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "2 3" && [string match "*Float*" $dtype]}
} {1}

# Test cases for camelCase alias
test tensor_sigmoid-3.1 {CamelCase alias syntax} {
    set t1 [torch::full {1} 0.0]
    set result [torch::tensorSigmoid $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.5) < 0.001}
} {1}

test tensor_sigmoid-3.2 {CamelCase with named parameters} {
    set t1 [torch::full {1} 3.0]
    set result [torch::tensorSigmoid -input $t1]
    set value [torch::tensor_item $result]
    # sigmoid(3) ≈ 0.9526
    expr {abs($value - 0.9526) < 0.01}
} {1}

# Error handling tests
test tensor_sigmoid-4.1 {Error: invalid tensor name (positional)} {
    catch {torch::tensor_sigmoid invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_sigmoid-4.2 {Error: invalid tensor name (named)} {
    catch {torch::tensor_sigmoid -input invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_sigmoid-4.3 {Error: missing required parameter} {
    catch {torch::tensor_sigmoid} result
    expr {[string match "*Usage*" $result] || [string match "*Required parameter*" $result]}
} {1}

test tensor_sigmoid-4.4 {Error: unknown parameter} {
    set t1 [torch::ones {1}]
    catch {torch::tensor_sigmoid -invalid_param $t1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test tensor_sigmoid-4.5 {Error: missing value for named parameter} {
    catch {torch::tensor_sigmoid -input} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

# Mathematical correctness tests
test tensor_sigmoid-5.1 {Mathematical properties: range [0,1]} {
    set t1 [torch::full {1} 10.0]
    set result_large [torch::tensor_sigmoid $t1]
    set value_large [torch::tensor_item $result_large]
    
    set t2 [torch::full {1} -10.0]
    set result_small [torch::tensor_sigmoid $t2]
    set value_small [torch::tensor_item $result_small]
    
    expr {$value_large < 1.0 && $value_large > 0.0 && $value_small < 1.0 && $value_small > 0.0}
} {1}

test tensor_sigmoid-5.2 {Mathematical properties: symmetry around 0.5} {
    set t1 [torch::full {1} 1.0]
    set result_pos [torch::tensor_sigmoid $t1]
    set value_pos [torch::tensor_item $result_pos]
    
    set t2 [torch::full {1} -1.0]
    set result_neg [torch::tensor_sigmoid $t2]
    set value_neg [torch::tensor_item $result_neg]
    
    # sigmoid(x) + sigmoid(-x) = 1.0
    expr {abs(($value_pos + $value_neg) - 1.0) < 0.001}
} {1}

test tensor_sigmoid-5.3 {Mathematical properties: monotonically increasing} {
    set t1 [torch::full {1} -1.0]
    set result1 [torch::tensor_sigmoid $t1]
    set value1 [torch::tensor_item $result1]
    
    set t2 [torch::full {1} 0.0]
    set result2 [torch::tensor_sigmoid $t2]
    set value2 [torch::tensor_item $result2]
    
    set t3 [torch::full {1} 1.0]
    set result3 [torch::tensor_sigmoid $t3]
    set value3 [torch::tensor_item $result3]
    
    expr {$value1 < $value2 && $value2 < $value3}
} {1}

# Data type consistency tests
test tensor_sigmoid-6.1 {Different data types: float64} {
    set t1 [torch::full {1} 1.0 float64]
    set result [torch::tensor_sigmoid $t1]
    set dtype [torch::tensor_dtype $result]
    set value [torch::tensor_item $result]
    expr {[string match "*Float64*" $dtype] && abs($value - 0.7311) < 0.01}
} {1}

# Integration tests with other commands
test tensor_sigmoid-7.1 {Integration with tensor creation} {
    set t1 [torch::full {1} 0.0]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.5) < 0.001}
} {1}

test tensor_sigmoid-7.2 {Chain operations: sigmoid then comparison} {
    set t1 [torch::full {1} 0.0]
    set sigmoid_result [torch::tensor_sigmoid $t1]
    set half [torch::full {1} 0.5]
    set diff [torch::tensor_sub $sigmoid_result $half]
    set diff_value [torch::tensor_item $diff]
    expr {abs($diff_value) < 0.001}
} {1}

# Syntax consistency tests
test tensor_sigmoid-8.1 {Both syntaxes produce same result} {
    set t1 [torch::full {1} 1.5]
    set result1 [torch::tensor_sigmoid $t1]
    set result2 [torch::tensor_sigmoid -input $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test tensor_sigmoid-8.2 {CamelCase alias produces same result} {
    set t1 [torch::full {1} -1.5]
    set result1 [torch::tensor_sigmoid $t1]
    set result2 [torch::tensorSigmoid $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {abs($value1 - $value2) < 0.000001}
} {1}

# Special mathematical cases
test tensor_sigmoid-9.1 {Edge case: very large positive values} {
    set t1 [torch::full {1} 100.0]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    # Due to floating-point precision, sigmoid(100) = 1.0
    expr {$value == 1.0}
} {1}

test tensor_sigmoid-9.2 {Edge case: very large negative values} {
    set t1 [torch::full {1} -100.0]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    # Due to floating-point precision, sigmoid(-100) = 0.0
    expr {$value == 0.0}
} {1}

test tensor_sigmoid-9.3 {Derivative relationship: sigmoid(x) * (1 - sigmoid(x))} {
    set t1 [torch::full {1} 1.0]
    set sigmoid_result [torch::tensor_sigmoid $t1]
    set sig_value [torch::tensor_item $sigmoid_result]
    
    # Mathematical property: derivative = sigmoid(x) * (1 - sigmoid(x))
    set expected_derivative [expr {$sig_value * (1.0 - $sig_value)}]
    
    # For sigmoid, derivative at x=1 ≈ 0.196611775
    expr {abs($expected_derivative - 0.1966) < 0.01}
} {1}

# Numerical stability tests
test tensor_sigmoid-10.1 {Numerical stability for small values} {
    set t1 [torch::full {1} 0.001]
    set result [torch::tensor_sigmoid $t1]
    set value [torch::tensor_item $result]
    # sigmoid(0.001) ≈ 0.50025
    expr {abs($value - 0.50025) < 0.001}
} {1}

test tensor_sigmoid-10.2 {Boundary behavior at typical neural network values} {
    # Test common activation function input ranges
    set inputs {-6.0 -3.0 -1.0 0.0 1.0 3.0 6.0}
    set all_valid 1
    
    foreach input $inputs {
        set t [torch::full {1} $input]
        set result [torch::tensor_sigmoid $t]
        set value [torch::tensor_item $result]
        
        # All sigmoid outputs should be in valid range [0,1]
        if {$value <= 0.0 || $value >= 1.0} {
            set all_valid 0
            break
        }
    }
    
    expr {$all_valid}
} {1}

cleanupTests 