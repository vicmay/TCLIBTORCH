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

# Helper to create a 1D tensor
proc create_1d_tensor {} {
    return [torch::tensor_create {1 2 3 4 5 6 7 8} float32 cpu true]
}

# Helper to create a 2D tensor
proc create_2d_tensor {} {
    return [torch::tensor_create {{1 2 3 4} {5 6 7 8} {9 10 11 12}} float32 cpu true]
}

# Test cases for positional syntax
test unfold-1.1 {Basic positional syntax - 1D tensor} {
    set t [create_1d_tensor]
    set unfold [torch::unfold $t 0 3 2]
    set shape [torch::tensor_shape $unfold]
    return $shape
} {3 3}

test unfold-1.2 {Positional syntax - 2D tensor} {
    set t [create_2d_tensor]
    set unfold [torch::unfold $t 1 2 1]
    set shape [torch::tensor_shape $unfold]
    return $shape
} {3 3 2}

# Test cases for named parameter syntax
test unfold-2.1 {Named parameter syntax - 1D tensor} {
    set t [create_1d_tensor]
    set unfold [torch::unfold -input $t -dimension 0 -size 3 -step 2]
    set shape [torch::tensor_shape $unfold]
    return $shape
} {3 3}

test unfold-2.2 {Named parameter syntax - 2D tensor} {
    set t [create_2d_tensor]
    set unfold [torch::unfold -input $t -dimension 1 -size 2 -step 1]
    set shape [torch::tensor_shape $unfold]
    return $shape
} {3 3 2}

test unfold-2.3 {Named parameter syntax with -tensor} {
    set t [create_1d_tensor]
    set unfold [torch::unfold -tensor $t -dimension 0 -size 3 -step 2]
    set shape [torch::tensor_shape $unfold]
    return $shape
} {3 3}

# Error handling tests
test unfold-3.1 {Error handling - missing input} {
    set result [catch {torch::unfold} error]
    return [list $result [string range $error 0 50]]
} {1 {Usage: torch::unfold input dimension size step | to}}

test unfold-3.2 {Error handling - invalid dimension} {
    set t [create_1d_tensor]
    set result [catch {torch::unfold $t foo 3 2} error]
    return $result
} {1}

test unfold-3.3 {Error handling - invalid size} {
    set t [create_1d_tensor]
    set result [catch {torch::unfold $t 0 foo 2} error]
    return $result
} {1}

test unfold-3.4 {Error handling - invalid step} {
    set t [create_1d_tensor]
    set result [catch {torch::unfold $t 0 3 foo} error]
    return $result
} {1}

test unfold-3.5 {Error handling - unknown named parameter} {
    set t [create_1d_tensor]
    set result [catch {torch::unfold -input $t -foo 0 -size 3 -step 2} error]
    return $result
} {1}

test unfold-3.6 {Error handling - missing value for parameter} {
    set t [create_1d_tensor]
    set result [catch {torch::unfold -input $t -dimension} error]
    return $result
} {1}

test unfold-3.7 {Error handling - invalid tensor name} {
    set result [catch {torch::unfold nonexistent_tensor 0 3 2} error]
    return $result
} {1}

# Mathematical consistency
test unfold-4.1 {Mathematical consistency between syntaxes} {
    set t [create_1d_tensor]
    set unfold1 [torch::unfold $t 0 3 2]
    set unfold2 [torch::unfold -input $t -dimension 0 -size 3 -step 2]
    set shape1 [torch::tensor_shape $unfold1]
    set shape2 [torch::tensor_shape $unfold2]
    return [list $shape1 $shape2]
} {{3 3} {3 3}}

# Edge cases
test unfold-5.1 {Edge case - size equals step} {
    set t [create_1d_tensor]
    set unfold [torch::unfold $t 0 2 2]
    set shape [torch::tensor_shape $unfold]
    return $shape
} {4 2}

test unfold-5.2 {Edge case - size larger than step} {
    set t [create_1d_tensor]
    set unfold [torch::unfold $t 0 4 1]
    set shape [torch::tensor_shape $unfold]
    return $shape
} {5 4}

cleanupTests 