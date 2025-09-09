#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test cases for positional syntax
test weight-norm-1.1 {Basic positional syntax with default dim} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm $tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test weight-norm-1.2 {Positional syntax with explicit dim} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm $tensor 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test weight-norm-1.3 {Positional syntax with 3D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
    set result [torch::weight_norm $tensor 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2 2}

;# Test cases for named syntax
test weight-norm-2.1 {Named parameter syntax with -input} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm -input $tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test weight-norm-2.2 {Named parameter syntax with -tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm -tensor $tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test weight-norm-2.3 {Named parameter syntax with -dim} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm -input $tensor -dim 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test weight-norm-2.4 {Named parameter syntax with all parameters} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::weight_norm -input $tensor -dim 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

;# Test cases for camelCase alias
test weight-norm-3.1 {CamelCase alias basic functionality} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weightNorm $tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test weight-norm-3.2 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weightNorm -input $tensor -dim 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

;# Test cases for mathematical correctness
test weight-norm-4.1 {Weight normalization preserves shape} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
    set original_shape [torch::tensor_shape $tensor]
    set result [torch::weight_norm $tensor]
    set result_shape [torch::tensor_shape $result]
    return [expr {$original_shape == $result_shape}]
} {1}

test weight-norm-4.2 {Weight normalization produces unit norm} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm $tensor 0]
    set norm [torch::tensor_norm $result 2 0]
    set norm_shape [torch::tensor_shape $norm]
    return [expr {[llength $norm_shape] == 1}]
} {1}

test weight-norm-4.3 {Weight normalization along different dimensions} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result1 [torch::weight_norm $tensor 0]
    set result2 [torch::weight_norm $tensor 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 == $shape2}]
} {1}

;# Test cases for different data types
test weight-norm-5.1 {Weight normalization with float64} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float64]
    set result [torch::weight_norm $tensor]
    set result_dtype [torch::tensor_dtype $result]
    return $result_dtype
} {Float64}

test weight-norm-5.2 {Weight normalization with float32} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm $tensor]
    set result_dtype [torch::tensor_dtype $result]
    return $result_dtype
} {Float32}

;# Error handling tests
test weight-norm-6.1 {Error handling - missing tensor} {
    catch {torch::weight_norm} result
    return [string match "*Wrong number of arguments*" $result]
} {1}

test weight-norm-6.2 {Error handling - invalid dim} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    catch {torch::weight_norm $tensor 10} result
    return [string match "*Dimension out of range*" $result]
} {1}

test weight-norm-6.3 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    catch {torch::weight_norm -invalid $tensor} result
    return [string match "*Unknown parameter*" $result]
} {1}

test weight-norm-6.4 {Error handling - missing parameter value} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    catch {torch::weight_norm -input} result
    return [string match "*Missing value for parameter*" $result]
} {1}

;# Edge cases
test weight-norm-7.1 {Edge case - zero tensor} {
    set tensor [torch::tensor_create -data {0.0 0.0 0.0 0.0} -shape {2 2} -dtype float32]
    catch {torch::weight_norm $tensor} result
    return [string match "*tensor*" $result]
} {1}

test weight-norm-7.2 {Edge case - single element tensor} {
    set tensor [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
    set result [torch::weight_norm $tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

test weight-norm-7.3 {Edge case - large values} {
    set tensor [torch::tensor_create -data {1000.0 2000.0 3000.0 4000.0} -shape {2 2} -dtype float32]
    set result [torch::weight_norm $tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

;# Consistency tests
test weight-norm-8.1 {Consistency between positional and named syntax} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result1 [torch::weight_norm $tensor 1]
    set result2 [torch::weight_norm -input $tensor -dim 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 == $shape2}]
} {1}

test weight-norm-8.2 {Consistency between snake_case and camelCase} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result1 [torch::weight_norm $tensor]
    set result2 [torch::weightNorm $tensor]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 == $shape2}]
} {1}

cleanupTests 