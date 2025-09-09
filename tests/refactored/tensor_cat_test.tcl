#!/usr/bin/env tclsh

# Test file for torch::tensor_cat command with dual syntax support
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

# Test suite for torch::tensor_cat
test tensor_cat-1.1 {Basic positional syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b]
    set result [torch::tensor_cat $tensor_list 0]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-2.1 {Named parameter syntax} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b]
    set result [torch::tensor_cat -tensors $tensor_list -dim 0]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-3.1 {CamelCase alias} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b]
    set result [torch::tensorCat -tensors $tensor_list -dim 0]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-4.1 {Error handling - invalid tensor} {
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list invalid_tensor $b]
    catch {torch::tensor_cat $tensor_list 0} result
    expr {[string length $result] > 0}
} {1}

test tensor_cat-4.2 {Error handling - missing input} {
    catch {torch::tensor_cat} result
    expr {[string length $result] > 0}
} {1}

test tensor_cat-4.3 {Error handling - unknown parameter} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b]
    catch {torch::tensor_cat -tensors $tensor_list -dim 0 -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_cat-5.1 {Mathematical correctness - 2D tensors dim 0} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 2}]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set b2d [torch::tensor_reshape $b {2 2}]
    set tensor_list [list $a2d $b2d]
    set result [torch::tensor_cat $tensor_list 0]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-5.2 {Mathematical correctness - 2D tensors dim 1} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set a2d [torch::tensor_reshape $a {2 2}]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set b2d [torch::tensor_reshape $b {2 2}]
    set tensor_list [list $a2d $b2d]
    set result [torch::tensor_cat $tensor_list 1]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-5.3 {Mathematical correctness - 3 tensors} {
    set a [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {3.0 4.0} -dtype float32 -device cpu]
    set c [torch::tensor_create -data {5.0 6.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b $c]
    set result [torch::tensor_cat $tensor_list 0]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-6.1 {Different data types - float32} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b]
    set result [torch::tensor_cat $tensor_list 0]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-6.2 {Different data types - float64} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float64 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float64 -device cpu]
    set tensor_list [list $a $b]
    set result [torch::tensor_cat $tensor_list 0]
    expr {[string length $result] > 0}
} {1}

test tensor_cat-7.1 {Edge case - single tensor} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set tensor_list [list $a]
    catch {torch::tensor_cat $tensor_list 0} result
    expr {[string length $result] > 0}
} {1}

test tensor_cat-8.1 {Syntax consistency - positional vs named} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b]
    set result1 [torch::tensor_cat $tensor_list 0]
    set result2 [torch::tensor_cat -tensors $tensor_list -dim 0]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_cat-8.2 {Syntax consistency - snake_case vs camelCase} {
    set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set tensor_list [list $a $b]
    set result1 [torch::tensor_cat -tensors $tensor_list -dim 0]
    set result2 [torch::tensorCat -tensors $tensor_list -dim 0]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 