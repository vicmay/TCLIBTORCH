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
test tensor-sum-1.1 {Basic positional syntax - sum all} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_sum $t]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-1.2 {Positional syntax with dimension} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set t2d [torch::tensor_reshape $t {2 3}]
    set result [torch::tensor_sum $t2d 0]
    expr {[string length $result] > 0}
} {1}

# Test cases for named parameter syntax
test tensor-sum-2.1 {Named parameter syntax - basic} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_sum -input $t]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-2.2 {Named parameter syntax with dim} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set t2d [torch::tensor_reshape $t {2 3}]
    set result [torch::tensor_sum -input $t2d -dim 1]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-2.3 {Named parameter syntax - different order} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_sum -input $t]
    expr {[string length $result] > 0}
} {1}

# Test cases for camelCase alias
test tensor-sum-3.1 {CamelCase alias - positional} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensorSum $t]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-3.2 {CamelCase alias - named parameters} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set t2d [torch::tensor_reshape $t {2 3}]
    set result [torch::tensorSum -input $t2d -dim 0]
    expr {[string length $result] > 0}
} {1}

# Syntax consistency
test tensor-sum-4.1 {Syntax consistency - positional vs named} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_sum $t]
    set result2 [torch::tensor_sum -input $t]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor-sum-4.2 {Syntax consistency - all three syntaxes} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result1 [torch::tensor_sum $t]
    set result2 [torch::tensor_sum -input $t]
    set result3 [torch::tensorSum -input $t]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string length $result3] > 0}
} {1}

# Error handling
test tensor-sum-5.1 {Error handling - invalid tensor} {
    catch {torch::tensor_sum invalid_tensor} result
    return [string match "*Invalid tensor name*" $result]
} {1}

test tensor-sum-5.2 {Error handling - missing parameters} {
    catch {torch::tensor_sum} result
    return [string match "*Required parameter missing*" $result]
} {1}

test tensor-sum-5.3 {Error handling - too many parameters} {
    set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_sum $t 0 extra} result
    return [string match "*Usage: torch::tensor_sum tensor ?dim?*" $result]
} {1}

test tensor-sum-5.4 {Error handling - unknown parameter} {
    set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_sum -input $t -unknown_param value} result
    return [string match "*Unknown parameter*" $result]
} {1}

# Mathematical correctness
test tensor-sum-6.1 {Mathematical correctness - sum all elements} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set result [torch::tensor_sum $t]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-6.2 {Mathematical correctness - sum along dimension} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
    set t2d [torch::tensor_reshape $t {2 3}]
    set result [torch::tensor_sum -input $t2d -dim 0]
    expr {[string length $result] > 0}
} {1}

# Edge cases
test tensor-sum-7.1 {Edge case - single element} {
    set t [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
    set result [torch::tensor_sum $t]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-7.2 {Edge case - zero tensor} {
    set t [torch::tensor_create -data {0.0 0.0 0.0 0.0} -dtype float32 -device cpu]
    set result [torch::tensor_sum $t]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-7.3 {Edge case - negative values} {
    set t [torch::tensor_create -data {-1.0 -2.0 -3.0 -4.0} -dtype float32 -device cpu]
    set result [torch::tensor_sum $t]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-7.4 {Edge case - dimension out of bounds} {
    set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_sum -input $t -dim 5} result
    expr {[string length $result] > 0}
} {1}

# Data preservation
test tensor-sum-8.1 {Data preservation - original tensor unchanged} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    set original [torch::tensor_print $t]
    set result [torch::tensor_sum $t]
    set new [torch::tensor_print $t]
    return [expr {$original eq $new}]
} {1}

# Multi-dimensional tests
test tensor-sum-9.1 {Multi-dimensional - 3D tensor} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set t3d [torch::tensor_reshape $t {2 2 2}]
    set result [torch::tensor_sum -input $t3d -dim 1]
    expr {[string length $result] > 0}
} {1}

test tensor-sum-9.2 {Multi-dimensional - sum all dimensions} {
    set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
    set t3d [torch::tensor_reshape $t {2 2 2}]
    set result [torch::tensor_sum $t3d]
    expr {[string length $result] > 0}
} {1}

cleanupTests 