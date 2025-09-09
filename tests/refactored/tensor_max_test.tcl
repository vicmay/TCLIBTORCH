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

# Helper function to create test tensors
proc create_test_tensor {name shape values} {
    set t [torch::tensor_create $values $shape]
    return $t
}

# Test cases for positional syntax
test tensor-max-1.1 {Basic positional syntax - max of entire tensor} {
    set t [create_test_tensor "test1" {3} {1 5 3}]
    set result [torch::tensor_max $t]
    set max_val [torch::tensor_item $result]
    expr {$max_val == 5}
} {1}

test tensor-max-1.2 {Positional syntax - max along dimension} {
    set t [create_test_tensor "test2" {2 3} {1 5 3 2 4 6}]
    set result [torch::tensor_max $t 0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test tensor-max-1.3 {Positional syntax - max along another dimension} {
    set t [create_test_tensor "test3" {2 3} {1 5 3 2 4 6}]
    set result [torch::tensor_max $t 1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

# Test cases for named parameter syntax
test tensor-max-2.1 {Named parameter syntax - max of entire tensor} {
    set t [create_test_tensor "test4" {4} {1 8 3 2}]
    set result [torch::tensor_max -input $t]
    set max_val [torch::tensor_item $result]
    expr {$max_val == 8}
} {1}

test tensor-max-2.2 {Named parameter syntax - max along dimension} {
    set t [create_test_tensor "test5" {2 2} {1 3 2 4}]
    set result [torch::tensor_max -input $t -dim 0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

test tensor-max-2.3 {Named parameter syntax - max along another dimension} {
    set t [create_test_tensor "test6" {2 2} {1 3 2 4}]
    set result [torch::tensor_max -input $t -dim 1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

# Test cases for camelCase alias
test tensor-max-3.1 {CamelCase alias - max of entire tensor} {
    set t [create_test_tensor "test7" {3} {1 7 3}]
    set result [torch::tensorMax $t]
    set max_val [torch::tensor_item $result]
    expr {$max_val == 7}
} {1}

test tensor-max-3.2 {CamelCase alias - named parameters} {
    set t [create_test_tensor "test8" {2 2} {1 9 2 4}]
    set result [torch::tensorMax -input $t -dim 0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

# Error handling tests
test tensor-max-4.1 {Error - missing input} {
    catch {torch::tensor_max} result
    set result
} {Input tensor is required}

test tensor-max-4.2 {Error - unknown parameter} {
    set t [create_test_tensor "test9" {2} {1 2}]
    catch {torch::tensor_max -foo $t} result
    set result
} {Unknown parameter: -foo}

test tensor-max-4.3 {Error - missing value for parameter} {
    catch {torch::tensor_max -input} result
    set result
} {Missing value for parameter}

test tensor-max-4.4 {Error - invalid tensor name} {
    catch {torch::tensor_max invalid_tensor} result
    set result
} {Invalid tensor name}

test tensor-max-4.5 {Error - invalid dimension} {
    set t [create_test_tensor "test10" {2} {1 2}]
    catch {torch::tensor_max $t 10} result
    expr {[string first "Dimension out of range" $result] >= 0}
} {1}

test tensor-max-4.6 {Error - too many positional arguments} {
    set t [create_test_tensor "test11" {2} {1 2}]
    catch {torch::tensor_max $t 0 extra} result
    set result
} {Invalid number of arguments}

# Edge cases
test tensor-max-5.1 {Edge case - single element tensor} {
    set t [create_test_tensor "test12" {1} {42}]
    set result [torch::tensor_max $t]
    set max_val [torch::tensor_item $result]
    expr {$max_val == 42}
} {1}

test tensor-max-5.2 {Edge case - all same values} {
    set t [create_test_tensor "test13" {3} {5 5 5}]
    set result [torch::tensor_max $t]
    set max_val [torch::tensor_item $result]
    expr {$max_val == 5}
} {1}

test tensor-max-5.3 {Edge case - negative values} {
    # Use a different approach for negative values - create with positive values first
    set t [torch::tensor_create {1 5 3} {3}]
    # Then subtract to make them negative
    set t_neg [torch::tensor_sub $t [torch::tensor_create {2 6 4} {3}]]
    set result [torch::tensor_max $t_neg]
    set max_val [torch::tensor_item $result]
    expr {$max_val == -1}
} {1}

# Mathematical correctness
test tensor-max-6.1 {Mathematical correctness - 2D tensor max along dim 0} {
    set t [torch::tensor_create {1 5 3 2 4 6} {2 3}]
    set result [torch::tensor_max $t 0]
    set vals [torch::tensor_print $result]
    set has_2 [expr {[string first "2" $vals] >= 0}]
    set has_5 [expr {[string first "5" $vals] >= 0}]
    set has_6 [expr {[string first "6" $vals] >= 0}]
    expr {$has_2 && $has_5 && $has_6}
} {1}

test tensor-max-6.2 {Mathematical correctness - 2D tensor max along dim 1} {
    set t [torch::tensor_create {1 5 3 2 4 6} {2 3}]
    set result [torch::tensor_max $t 1]
    set vals [torch::tensor_print $result]
    set has_5 [expr {[string first "5" $vals] >= 0}]
    set has_6 [expr {[string first "6" $vals] >= 0}]
    expr {$has_5 && $has_6}
} {1}

# Syntax consistency
test tensor-max-7.1 {Syntax consistency - positional vs named} {
    set t [create_test_tensor "test17" {2} {1 3}]
    set result1 [torch::tensor_max $t]
    set result2 [torch::tensor_max -input $t]
    set val1 [torch::tensor_item $result1]
    set val2 [torch::tensor_item $result2]
    expr {$val1 == $val2}
} {1}

test tensor-max-7.2 {Syntax consistency - snake_case vs camelCase} {
    set t [create_test_tensor "test18" {2} {1 4}]
    set result1 [torch::tensor_max $t]
    set result2 [torch::tensorMax $t]
    set val1 [torch::tensor_item $result1]
    set val2 [torch::tensor_item $result2]
    expr {$val1 == $val2}
} {1}

# Data type support
test tensor-max-6.3 {Data type support - float32} {
    set t [create_test_tensor "test19" {3} {1.5 3.7 2.1}]
    set result [torch::tensor_max $t]
    set max_val [torch::tensor_item $result]
    expr {abs($max_val - 3.7) < 0.001}
} {1}

cleanupTests 