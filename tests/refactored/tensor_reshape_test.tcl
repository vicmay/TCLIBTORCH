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

# Test cases for positional syntax (backward compatibility)
test tensor-reshape-1.1 {Basic positional syntax - 1D to 2D} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_reshape $t {2 3}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test tensor-reshape-1.2 {Basic positional syntax - 2D to 1D} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set t2d [torch::tensor_reshape $t {2 3}]
    set result [torch::tensor_reshape $t2d {6}]
    set shape [torch::tensor_shape $result]
    return $shape
} {6}

test tensor-reshape-1.3 {Basic positional syntax - 3D reshape} {
    set t [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_reshape $t {2 2 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2 2}

test tensor-reshape-1.4 {Basic positional syntax - inferred dimension} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_reshape $t {-1 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 2}

test tensor-reshape-1.5 {Basic positional syntax - inferred dimension 2} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_reshape $t {2 -1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

# Test cases for named parameter syntax
test tensor-reshape-2.1 {Named parameter syntax - basic} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_reshape -input $t -shape {3 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 2}

test tensor-reshape-2.2 {Named parameter syntax - different order} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_reshape -shape {2 3} -input $t]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test tensor-reshape-2.3 {Named parameter syntax - with inferred dimension} {
    set t [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_reshape -input $t -shape {2 -1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 4}

# Test cases for camelCase alias
test tensor-reshape-3.1 {CamelCase alias - basic} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensorReshape $t {2 3}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test tensor-reshape-3.2 {CamelCase alias - named parameters} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensorReshape -input $t -shape {3 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 2}

# Test cases for syntax consistency
test tensor-reshape-4.1 {Syntax consistency - positional vs named} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result1 [torch::tensor_reshape $t {2 3}]
    set result2 [torch::tensor_reshape -input $t -shape {2 3}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 eq $shape2}]
} {1}

test tensor-reshape-4.2 {Syntax consistency - all three syntaxes} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result1 [torch::tensor_reshape $t {2 3}]
    set result2 [torch::tensor_reshape -input $t -shape {2 3}]
    set result3 [torch::tensorReshape $t {2 3}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    return [expr {$shape1 eq $shape2 && $shape2 eq $shape3}]
} {1}

# Error handling tests
test tensor-reshape-5.1 {Error handling - invalid tensor} {
    catch {torch::tensor_reshape invalid_tensor {2 3}} result
    return [string match "*Invalid tensor name*" $result]
} {1}

test tensor-reshape-5.2 {Error handling - missing parameters} {
    set t [torch::tensor_create {1 2 3 4}]
    catch {torch::tensor_reshape $t} result
    puts "Error message: $result"
    return [string match "*Missing value for parameter*" $result]
} {1}

test tensor-reshape-5.3 {Error handling - too many parameters} {
    set t [torch::tensor_create {1 2 3 4}]
    catch {torch::tensor_reshape $t {2 2} extra} result
    return [string match "*Usage*" $result]
} {1}

test tensor-reshape-5.4 {Error handling - invalid shape} {
    set t [torch::tensor_create {1 2 3 4}]
    catch {torch::tensor_reshape $t {2 3}} result
    return [string match "*size*" $result]
} {1}

test tensor-reshape-5.5 {Error handling - named syntax missing value} {
    set t [torch::tensor_create {1 2 3 4}]
    catch {torch::tensor_reshape -input $t -shape} result
    return [string match "*Missing value*" $result]
} {1}

test tensor-reshape-5.6 {Error handling - unknown parameter} {
    set t [torch::tensor_create {1 2 3 4}]
    catch {torch::tensor_reshape -input $t -unknown {2 2}} result
    return [string match "*Unknown parameter*" $result]
} {1}

# Edge cases and special shapes
test tensor-reshape-6.1 {Edge case - scalar to 1D} {
    set t [torch::tensor_create 42]
    set result [torch::tensor_reshape $t {1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

test tensor-reshape-6.2 {Edge case - 1D to scalar} {
    set t [torch::tensor_create {42}]
    set result [torch::tensor_reshape $t {1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

test tensor-reshape-6.3 {Edge case - empty tensor} {
    set t [torch::tensor_create {}]
    set result [torch::tensor_reshape $t {0}]
    set shape [torch::tensor_shape $result]
    return $shape
} {0}

test tensor-reshape-6.4 {Edge case - multiple inferred dimensions} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    catch {torch::tensor_reshape $t {-1 -1}} result
    return [string match "*inferred*" $result]
} {1}

# Data preservation tests
test tensor-reshape-7.1 {Data preservation - values unchanged} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_reshape $t {2 3}]
    set values [torch::tensor_print $result]
    return [string match "*1*2*3*4*5*6*" $values]
} {1}

test tensor-reshape-7.2 {Data preservation - reshape back} {
    set t [torch::tensor_create {1 2 3 4 5 6}]
    set reshaped [torch::tensor_reshape $t {2 3}]
    set back [torch::tensor_reshape $reshaped {6}]
    set values [torch::tensor_print $back]
    return [string match "*1*2*3*4*5*6*" $values]
} {1}

# Complex shapes
test tensor-reshape-8.1 {Complex shape - 4D tensor} {
    set t [torch::tensor_create {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}]
    set result [torch::tensor_reshape $t {2 2 2 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2 2 2}

test tensor-reshape-8.2 {Complex shape - mixed inferred dimensions} {
    set t [torch::tensor_create {1 2 3 4 5 6 7 8 9 10 11 12}]
    set result [torch::tensor_reshape $t {2 -1 2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3 2}

cleanupTests 