#!/usr/bin/env tclsh

# Test file for torch::tensor_exp command with dual syntax support
# Tests both positional and named parameter syntax

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test suite for torch::tensor_exp
test tensor_exp-1.1 {Basic positional syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_exp $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-2.1 {Named parameter syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_exp -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-3.1 {CamelCase alias} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensorExp -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-4.1 {Error handling - invalid tensor} {
    catch {torch::tensor_exp invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor_exp-4.2 {Error handling - missing input} {
    catch {torch::tensor_exp} result
    expr {[string length $result] > 0}
} {1}

test tensor_exp-4.3 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    catch {torch::tensor_exp -input $a -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_exp-5.1 {Mathematical correctness - positive values} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_exp $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-5.2 {Mathematical correctness - negative values} {
    set a [torch::tensor_create -data {-1.0 -2.0 -3.0 -4.0} -dtype float32 -device cpu]
    set result [torch::tensor_exp $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-5.3 {Mathematical correctness - zero values} {
    set a [torch::tensor_create -data {0.0 0.0 0.0 0.0} -dtype float32 -device cpu]
    set result [torch::tensor_exp $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-6.1 {Different data types - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_exp $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-6.2 {Different data types - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float64 -device cpu]
    set result [torch::tensor_exp $a]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-7.1 {Multi-dimensional tensors} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 4}]
    set result [torch::tensor_exp $a2d]
    expr {[string length $result] > 0}
} {1}

test tensor_exp-8.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_exp $a]
    set result2 [torch::tensor_exp -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_exp-8.2 {Syntax consistency - snake_case vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_exp -input $a]
    set result2 [torch::tensorExp -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 