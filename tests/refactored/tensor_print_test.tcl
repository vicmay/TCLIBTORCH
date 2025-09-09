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

# Test cases for positional syntax
test tensor-print-1.1 {Basic positional syntax with simple tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 3}]
    set result [torch::tensor_print $reshaped]
    # Should return a string representation of the tensor
    expr {[string length $result] > 0}
} {1}

test tensor-print-1.2 {Positional syntax with zeros tensor} {
    set tensor [torch::zeros -shape {2 2} -dtype float32]
    set result [torch::tensor_print $tensor]
    # Should return a string representation of the tensor
    expr {[string length $result] > 0}
} {1}

test tensor-print-1.3 {Positional syntax with ones tensor} {
    set tensor [torch::ones -shape {3 3} -dtype float32]
    set result [torch::tensor_print $tensor]
    # Should return a string representation of the tensor
    expr {[string length $result] > 0}
} {1}

# Test cases for named syntax
test tensor-print-2.1 {Named parameter syntax with simple tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result [torch::tensor_print -input $reshaped]
    # Should return a string representation of the tensor
    expr {[string length $result] > 0}
} {1}

test tensor-print-2.2 {Named syntax with random tensor} {
    set tensor [torch::tensor_randn {2 3} float32]
    set result [torch::tensor_print -input $tensor]
    # Should return a string representation of the tensor
    expr {[string length $result] > 0}
} {1}

test tensor-print-2.3 {Named syntax with different data types} {
    set tensor1 [torch::tensor_create -data {1 2 3 4} -dtype int32]
    set tensor2 [torch::tensor_create -data {1.5 2.5 3.5 4.5} -dtype float64]
    set result1 [torch::tensor_print -input $tensor1]
    set result2 [torch::tensor_print -input $tensor2]
    # Both should return string representations
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Test cases for camelCase alias
test tensor-print-3.1 {CamelCase alias tensorPrint} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result [torch::tensorPrint $reshaped]
    # Should return a string representation of the tensor
    expr {[string length $result] > 0}
} {1}

test tensor-print-3.2 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {3 2}]
    set result [torch::tensorPrint -input $reshaped]
    # Should return a string representation of the tensor
    expr {[string length $result] > 0}
} {1}

# Error handling tests
test tensor-print-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_print nonexistent_tensor} result
    set result
} {Invalid tensor name}

test tensor-print-4.2 {Error handling - missing arguments} {
    catch {torch::tensor_print} result
    set result
} {Input tensor is required}

test tensor-print-4.3 {Error handling - too many arguments} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    catch {torch::tensor_print $reshaped extra_arg} result
    set result
} {Invalid number of arguments}

test tensor-print-4.4 {Error handling - invalid named parameter} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    catch {torch::tensor_print -invalid_param $reshaped} result
    set result
} {Unknown parameter: -invalid_param}

test tensor-print-4.5 {Error handling - missing value for parameter} {
    catch {torch::tensor_print -input} result
    set result
} {Missing value for parameter}

# Edge cases and different data types
test tensor-print-5.1 {Edge case - empty tensor} {
    set tensor [torch::tensor_create -data {} -dtype float32]
    set result [torch::tensor_print $tensor]
    # Should return a string representation even for empty tensor
    expr {[string length $result] > 0}
} {1}

test tensor-print-5.2 {Edge case - single element tensor} {
    set tensor [torch::tensor_create -data 42.0 -dtype float32]
    set result [torch::tensor_print $tensor]
    # Should return a string representation
    expr {[string length $result] > 0}
} {1}

test tensor-print-5.3 {Edge case - large tensor} {
    set data {}
    for {set i 0} {$i < 100} {incr i} {
        lappend data 1.0
    }
    set tensor [torch::tensor_create -data $data -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {10 10}]
    set result [torch::tensor_print $reshaped]
    # Should return a string representation
    expr {[string length $result] > 0}
} {1}

test tensor-print-5.4 {Different data types - int tensor} {
    set tensor [torch::tensor_create -data {1 2 3 4} -dtype int32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result [torch::tensor_print $reshaped]
    # Should return a string representation
    expr {[string length $result] > 0}
} {1}

test tensor-print-5.5 {Different data types - double tensor} {
    set tensor [torch::tensor_create -data {1.5 2.5 3.5 4.5} -dtype float64]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result [torch::tensor_print $reshaped]
    # Should return a string representation
    expr {[string length $result] > 0}
} {1}

# Syntax consistency tests
test tensor-print-6.1 {Syntax consistency - positional vs named} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result1 [torch::tensor_print $reshaped]
    set result2 [torch::tensor_print -input $reshaped]
    # Both should return the same string representation
    string equal $result1 $result2
} {1}

test tensor-print-6.2 {Syntax consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result1 [torch::tensor_print $reshaped]
    set result2 [torch::tensorPrint $reshaped]
    # Both should return the same string representation
    string equal $result1 $result2
} {1}

test tensor-print-6.3 {Syntax consistency - all three syntaxes} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result1 [torch::tensor_print $reshaped]
    set result2 [torch::tensor_print -input $reshaped]
    set result3 [torch::tensorPrint $reshaped]
    # All should return the same string representation
    expr {[string equal $result1 $result2] && [string equal $result2 $result3]}
} {1}

# Mathematical correctness tests
test tensor-print-7.1 {Mathematical correctness - verify tensor content} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result [torch::tensor_print $reshaped]
    # The string should contain the tensor values
    expr {[string first "1" $result] >= 0 && [string first "2" $result] >= 0 && [string first "3" $result] >= 0 && [string first "4" $result] >= 0}
} {1}

test tensor-print-7.2 {Mathematical correctness - verify tensor shape} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {3 2}]
    set result [torch::tensor_print $reshaped]
    # The string should contain tensor shape information
    expr {[string first "3" $result] >= 0 && [string first "2" $result] >= 0}
} {1}

# Device independence tests
test tensor-print-8.1 {Device independence - CPU tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set result [torch::tensor_print $reshaped]
    # Should work on CPU tensor
    expr {[string length $result] > 0}
} {1}

# Cleanup
cleanupTests 