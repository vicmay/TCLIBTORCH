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

# Helper to create a 2D tensor
proc create_2d_tensor {} {
    return [torch::tensor_create {{1 2 3} {4 5 6}} float32 cpu true]
}

# Test cases for positional syntax
test unsqueezeMultiple-1.1 {Basic positional syntax} {
    set t [create_2d_tensor]
    set result [torch::unsqueeze_multiple $t {0 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {1 2 3 1}

test unsqueezeMultiple-1.2 {Positional syntax with single dimension} {
    set t [create_2d_tensor]
    set result [torch::unsqueeze_multiple $t {1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 1 3}

# Test cases for named parameter syntax
test unsqueezeMultiple-2.1 {Named parameter syntax} {
    set t [create_2d_tensor]
    set result [torch::unsqueeze_multiple -tensor $t -dims {0 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {1 2 3 1}

test unsqueezeMultiple-2.2 {Named parameter syntax with single dimension} {
    set t [create_2d_tensor]
    set result [torch::unsqueeze_multiple -tensor $t -dims {1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 1 3}

# Test cases for camelCase alias
test unsqueezeMultiple-3.1 {CamelCase alias - positional syntax} {
    set t [create_2d_tensor]
    set result [torch::unsqueezeMultiple $t {0 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {1 2 3 1}

test unsqueezeMultiple-3.2 {CamelCase alias - named parameter syntax} {
    set t [create_2d_tensor]
    set result [torch::unsqueezeMultiple -tensor $t -dims {0 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {1 2 3 1}

# Error handling tests
test unsqueezeMultiple-4.1 {Error handling - missing parameters} {
    set result [catch {torch::unsqueeze_multiple} error]
    return [list $result [string range $error 0 50]]
} {1 {Required parameters missing: tensor, dims}}

test unsqueezeMultiple-4.2 {Error handling - missing tensor} {
    set result [catch {torch::unsqueeze_multiple -dims {0 1}} error]
    return $result
} {1}

test unsqueezeMultiple-4.3 {Error handling - missing dims} {
    set t [create_2d_tensor]
    set result [catch {torch::unsqueeze_multiple -tensor $t} error]
    return $result
} {1}

test unsqueezeMultiple-4.4 {Error handling - unknown named parameter} {
    set t [create_2d_tensor]
    set result [catch {torch::unsqueeze_multiple -tensor $t -foo {0 1}} error]
    return $result
} {1}

test unsqueezeMultiple-4.5 {Error handling - missing value for parameter} {
    set t [create_2d_tensor]
    set result [catch {torch::unsqueeze_multiple -tensor $t -dims} error]
    return $result
} {1}

test unsqueezeMultiple-4.6 {Error handling - invalid tensor name} {
    set result [catch {torch::unsqueeze_multiple nonexistent_tensor {0 1}} error]
    return $result
} {1}

# Mathematical consistency
test unsqueezeMultiple-5.1 {Mathematical consistency between syntaxes} {
    set t [create_2d_tensor]
    set result1 [torch::unsqueeze_multiple $t {0 2}]
    set result2 [torch::unsqueeze_multiple -tensor $t -dims {0 2}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [list $shape1 $shape2]
} {{1 2 3 1} {1 2 3 1}}

# Edge cases
test unsqueezeMultiple-6.1 {Edge case - empty dims list} {
    set t [create_2d_tensor]
    set result [catch {torch::unsqueeze_multiple $t {}} error]
    return $result
} {1}

test unsqueezeMultiple-6.2 {Edge case - negative dimensions} {
    set t [create_2d_tensor]
    set result [torch::unsqueeze_multiple $t {-1 -3}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 1 3 1}

test unsqueezeMultiple-6.3 {Edge case - out of bounds dimensions} {
    set t [create_2d_tensor]
    set result [catch {torch::unsqueeze_multiple $t {5 6}} error]
    return $result
} {1}

cleanupTests 