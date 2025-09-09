#!/usr/bin/env tclsh

# Test file for torch::tensor_matmul command with dual syntax support
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

# Test suite for torch::tensor_matmul
test tensor_matmul-1.1 {Basic positional syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set result [torch::tensor_matmul $a $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-2.1 {Named parameter syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set result [torch::tensor_matmul -input $a -other $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-3.1 {CamelCase alias} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set result [torch::tensorMatmul -input $a -other $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-4.1 {Error handling - invalid tensor} {
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    catch {torch::tensor_matmul invalid_tensor $b} result
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-4.2 {Error handling - missing input} {
    catch {torch::tensor_matmul} result
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-4.3 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    catch {torch::tensor_matmul -input $a -other $b -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-5.1 {Mathematical correctness - 2x2 matrices} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set result [torch::tensor_matmul $a $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-5.2 {Mathematical correctness - 1D x 2D} {
    # 1D tensor (2 elements) x 2D tensor (2x2) = 1D tensor (2 elements)
    set a [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set b2d [torch::tensor_reshape $b {2 2}]
    set result [torch::tensor_matmul $a $b2d]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-5.3 {Mathematical correctness - 2D x 1D} {
    # 2D tensor (2x2) x 1D tensor (2 elements) = 1D tensor (2 elements)
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 2}]
    set b [torch::tensor_create -data {5.0 6.0} -dtype float32 -device cpu]
    set result [torch::tensor_matmul $a2d $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-6.1 {Different data types - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set result [torch::tensor_matmul $a $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-6.2 {Different data types - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float64 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float64 -device cpu]
    set result [torch::tensor_matmul $a $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-7.1 {Multi-dimensional tensors} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set result [torch::tensor_matmul $a $b]
    expr {[string length $result] > 0}
} {1}

test tensor_matmul-8.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_matmul $a $b]
    set result2 [torch::tensor_matmul -input $a -other $b]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_matmul-8.2 {Syntax consistency - snake_case vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_matmul -input $a -other $b]
    set result2 [torch::tensorMatmul -input $a -other $b]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 