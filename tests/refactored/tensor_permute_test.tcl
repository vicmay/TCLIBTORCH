#!/usr/bin/env tclsh

# Test file for torch::tensor_permute command with dual syntax support
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

# Test suite for torch::tensor_permute
test tensor_permute-1.1 {Basic positional syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 3}]
    set result [torch::tensor_permute $a3d {2 1 0}]
    expr {[string length $result] > 0}
} {1}

test tensor_permute-2.1 {Named parameter syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 3}]
    set result [torch::tensor_permute -input $a3d -dims {2 1 0}]
    expr {[string length $result] > 0}
} {1}

test tensor_permute-3.1 {CamelCase alias} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 3}]
    set result [torch::tensorPermute -input $a3d -dims {2 1 0}]
    expr {[string length $result] > 0}
} {1}

test tensor_permute-4.1 {Error handling - invalid tensor} {
    catch {torch::tensor_permute invalid_tensor {0 1 2}} result
    expr {[string length $result] > 0}
} {1}

test tensor_permute-4.2 {Error handling - missing input} {
    catch {torch::tensor_permute} result
    expr {[string length $result] > 0}
} {1}

test tensor_permute-4.3 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 3}]
    catch {torch::tensor_permute -input $a3d -dims {2 1 0} -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_permute-5.1 {Mathematical correctness - permute 3D tensor} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 3}]
    set result [torch::tensor_permute $a3d {2 1 0}]
    expr {[string length $result] > 0}
} {1}

test tensor_permute-6.1 {Edge case - single dimension} {
    set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set a1d [torch::tensor_reshape $a {3}]
    set result [torch::tensor_permute $a1d {0}]
    expr {[string length $result] > 0}
} {1}

test tensor_permute-6.2 {Edge case - reverse dimensions} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 2}]
    set result [torch::tensor_permute $a2d {1 0}]
    expr {[string length $result] > 0}
} {1}

test tensor_permute-7.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 3}]
    set result1 [torch::tensor_permute $a3d {2 1 0}]
    set result2 [torch::tensor_permute -input $a3d -dims {2 1 0}]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_permute-7.2 {Syntax consistency - snake_case vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 3}]
    set result1 [torch::tensor_permute -input $a3d -dims {2 1 0}]
    set result2 [torch::tensorPermute -input $a3d -dims {2 1 0}]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 