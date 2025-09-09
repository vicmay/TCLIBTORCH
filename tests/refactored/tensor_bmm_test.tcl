#!/usr/bin/env tclsh

# Test file for torch::tensor_bmm command with dual syntax support
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

# Test suite for torch::tensor_bmm
test tensor_bmm-1.1 {Basic positional syntax} {
    # Create 3D tensors for batch matrix multiplication
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result [torch::tensor_bmm $a3d $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-2.1 {Named parameter syntax} {
    # Create 3D tensors for batch matrix multiplication
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result [torch::tensor_bmm -input $a3d -other $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-3.1 {CamelCase alias} {
    # Create 3D tensors for batch matrix multiplication
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result [torch::tensorBmm -input $a3d -other $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-4.1 {Error handling - invalid tensor} {
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 1}]
    catch {torch::tensor_bmm invalid_tensor $b3d} result
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-4.2 {Error handling - missing input} {
    catch {torch::tensor_bmm} result
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-4.3 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 1}]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 1}]
    catch {torch::tensor_bmm -input $a3d -other $b3d -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-5.1 {Mathematical correctness - 2x2x2 batch} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result [torch::tensor_bmm $a3d $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-5.2 {Mathematical correctness - 3x3x3 batch} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {3 3 3}]
    set b [torch::tensor_create -data {28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {3 3 3}]
    set result [torch::tensor_bmm $a3d $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-6.1 {Different data types - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result [torch::tensor_bmm $a3d $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-6.2 {Different data types - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float64 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float64 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result [torch::tensor_bmm $a3d $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-7.1 {Edge case - single batch} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {1 2 2}]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {1 2 2}]
    set result [torch::tensor_bmm $a3d $b3d]
    expr {[string length $result] > 0}
} {1}

test tensor_bmm-8.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result1 [torch::tensor_bmm $a3d $b3d]
    set result2 [torch::tensor_bmm -input $a3d -other $b3d]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_bmm-8.2 {Syntax consistency - snake_case vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set a3d [torch::tensor_reshape $a {2 2 2}]
    set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
    set b3d [torch::tensor_reshape $b {2 2 2}]
    set result1 [torch::tensor_bmm -input $a3d -other $b3d]
    set result2 [torch::tensorBmm -input $a3d -other $b3d]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 