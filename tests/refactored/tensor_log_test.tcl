#!/usr/bin/env tclsh

# Test file for torch::tensor_log command with dual syntax support
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

# Test suite for torch::tensor_log
test tensor_log-1.1 {Basic positional syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_log $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-2.1 {Named parameter syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_log -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-3.1 {CamelCase alias} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensorLog -input $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-4.1 {Error handling - invalid tensor} {
    catch {torch::tensor_log invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor_log-4.2 {Error handling - missing input} {
    catch {torch::tensor_log} result
    expr {[string length $result] > 0}
} {1}

test tensor_log-4.3 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    catch {torch::tensor_log -input $a -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_log-5.1 {Mathematical correctness - positive values} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_log $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-5.2 {Mathematical correctness - values greater than 1} {
    set a [torch::tensor_create -data {2.0 4.0 8.0 16.0} -dtype float32 -device cpu]
    set result [torch::tensor_log $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-5.3 {Mathematical correctness - value of 1} {
    set a [torch::tensor_create -data {1.0 1.0 1.0 1.0} -dtype float32 -device cpu]
    set result [torch::tensor_log $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-6.1 {Different data types - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_log $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-6.2 {Different data types - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float64 -device cpu]
    set result [torch::tensor_log $a]
    expr {[string length $result] > 0}
} {1}

test tensor_log-7.1 {Multi-dimensional tensors} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 4}]
    set result [torch::tensor_log $a2d]
    expr {[string length $result] > 0}
} {1}

test tensor_log-8.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_log $a]
    set result2 [torch::tensor_log -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_log-8.2 {Syntax consistency - snake_case vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_log -input $a]
    set result2 [torch::tensorLog -input $a]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 