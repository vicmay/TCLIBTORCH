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
test encoder-layer-1.1 {Basic positional syntax} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformer_encoder_layer $src 4 2 8 0.1]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test encoder-layer-1.2 {Positional syntax with different dropout} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformer_encoder_layer $src 4 2 8 0.5]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

# Test cases for named parameter syntax
test encoder-layer-2.1 {Named parameter syntax} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformer_encoder_layer -src $src -dModel 4 -nhead 2 -dimFeedforward 8 -dropout 0.1]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test encoder-layer-2.2 {Named parameter syntax with different dropout} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformer_encoder_layer -src $src -dModel 4 -nhead 2 -dimFeedforward 8 -dropout 0.5]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

# Test cases for camelCase alias
test encoder-layer-3.1 {CamelCase alias} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformerEncoderLayer -src $src -dModel 4 -nhead 2 -dimFeedforward 8 -dropout 0.1]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test encoder-layer-3.2 {CamelCase alias with positional syntax} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformerEncoderLayer $src 4 2 8 0.1]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

# Error handling tests
test encoder-layer-4.1 {Error handling - missing parameters} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder_layer $src 4 2} error]
    return [list $result $error]
} {1 {Usage: torch::transformer_encoder_layer src d_model nhead dim_feedforward dropout}}

test encoder-layer-4.2 {Error handling - invalid dModel} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder_layer $src 0 2 8 0.1} error]
    expr {$result == 1 && [string match {*Invalid parameters: src tensor must be defined, dModel, nhead, dimFeedforward, and dropout must be valid*} $error]}
} {1}

test encoder-layer-4.3 {Error handling - invalid named parameter} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder_layer -src $src -invalid 4} error]
    return [list $result [string range $error 0 50]]
} {1 {Unknown parameter: -invalid}}

test encoder-layer-4.4 {Error handling - missing value for parameter} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder_layer -src $src -dModel} error]
    return [list $result [string range $error 0 50]]
} {1 {Missing value for parameter}}

# Mathematical consistency
test encoder-layer-5.1 {Mathematical consistency between syntaxes} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result1 [torch::transformer_encoder_layer $src 4 2 8 0.1]
    set result2 [torch::transformer_encoder_layer -src $src -dModel 4 -nhead 2 -dimFeedforward 8 -dropout 0.1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [list $shape1 $shape2]
} {4 4}

# Data type test
test encoder-layer-6.1 {Different data types} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float64 cpu true]
    set result [torch::transformer_encoder_layer $src 4 2 8 0.1]
    set dtype [torch::tensor_dtype $result]
    return $dtype
} {Float64}

cleanupTests 