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
test uniform-1.1 {Basic positional syntax} {
    set uniform [torch::uniform {2 3} 0.0 1.0]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {2 3}

test uniform-1.2 {Positional syntax with dtype} {
    set uniform [torch::uniform {2 2} 0.0 1.0 float64]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {2 2}

test uniform-1.3 {Positional syntax with custom range} {
    set uniform [torch::uniform {3 3} -5.0 5.0]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {3 3}

# Test cases for named parameter syntax
test uniform-2.1 {Named parameter syntax} {
    set uniform [torch::uniform -size {2 3} -low 0.0 -high 1.0]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {2 3}

test uniform-2.2 {Named parameter syntax with dtype} {
    set uniform [torch::uniform -size {2 2} -low 0.0 -high 1.0 -dtype float64]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {2 2}

test uniform-2.3 {Named parameter syntax with device} {
    set uniform [torch::uniform -size {2 2} -low 0.0 -high 1.0 -device cpu]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {2 2}

test uniform-2.4 {Named parameter syntax with custom range} {
    set uniform [torch::uniform -size {3 3} -low -10.0 -high 10.0]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {3 3}

# Error handling tests
test uniform-3.1 {Error handling - missing parameters} {
    set result [catch {torch::uniform} error]
    return [list $result [string range $error 0 50]]
} {1 {Usage: torch::uniform size low high ?dtype? ?device}}

test uniform-3.2 {Error handling - invalid size} {
    set result [catch {torch::uniform foo 0.0 1.0} error]
    return $result
} {1}

test uniform-3.3 {Error handling - invalid low} {
    set result [catch {torch::uniform {2 2} foo 1.0} error]
    return $result
} {1}

test uniform-3.4 {Error handling - invalid high} {
    set result [catch {torch::uniform {2 2} 0.0 foo} error]
    return $result
} {1}

test uniform-3.5 {Error handling - low >= high} {
    set result [catch {torch::uniform {2 2} 1.0 0.0} error]
    return $result
} {1}

test uniform-3.6 {Error handling - unknown named parameter} {
    set result [catch {torch::uniform -size {2 2} -foo 0.0 -high 1.0} error]
    return $result
} {1}

test uniform-3.7 {Error handling - missing value for parameter} {
    set result [catch {torch::uniform -size {2 2} -low} error]
    return $result
} {1}

test uniform-3.8 {Error handling - invalid dtype} {
    set result [catch {torch::uniform {2 2} 0.0 1.0 invalid_dtype} error]
    return $result
} {1}

# Mathematical consistency
test uniform-4.1 {Mathematical consistency between syntaxes} {
    set uniform1 [torch::uniform {2 2} 0.0 1.0]
    set uniform2 [torch::uniform -size {2 2} -low 0.0 -high 1.0]
    set shape1 [torch::tensor_shape $uniform1]
    set shape2 [torch::tensor_shape $uniform2]
    return [list $shape1 $shape2]
} {{2 2} {2 2}}

# Edge cases
test uniform-5.1 {Edge case - negative range} {
    set uniform [torch::uniform {2 2} -1.0 -0.5]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {2 2}

test uniform-5.2 {Edge case - large range} {
    set uniform [torch::uniform {1 1} -1000.0 1000.0]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {1 1}

test uniform-5.3 {Edge case - small range} {
    set uniform [torch::uniform {2 2} 0.0 0.001]
    set shape [torch::tensor_shape $uniform]
    return $shape
} {2 2}

cleanupTests 