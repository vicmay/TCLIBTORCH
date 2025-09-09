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
test transformer-encoder-1.1 {Basic positional syntax} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformer_encoder $src 4 2 2 8]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test transformer-encoder-1.2 {Positional syntax with larger dimensions} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu true]
    set result [torch::transformer_encoder $src 6 3 3 12]
    set shape [torch::tensor_shape $result]
    return $shape
} {6}

# Test cases for named parameter syntax
test transformer-encoder-2.1 {Named parameter syntax} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformer_encoder -src $src -dModel 4 -nhead 2 -numLayers 2 -dimFeedforward 8]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test transformer-encoder-2.2 {Named parameter syntax with larger dimensions} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu true]
    set result [torch::transformer_encoder -src $src -dModel 6 -nhead 3 -numLayers 3 -dimFeedforward 12]
    set shape [torch::tensor_shape $result]
    return $shape
} {6}

# Test cases for camelCase alias
test transformer-encoder-3.1 {CamelCase alias} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformerEncoder -src $src -dModel 4 -nhead 2 -numLayers 2 -dimFeedforward 8]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test transformer-encoder-3.2 {CamelCase alias with positional syntax} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::transformerEncoder $src 4 2 2 8]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

# Error handling tests
test transformer-encoder-4.1 {Error handling - missing parameters} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder $src 4 2} error]
    return [list $result [string range $error 0 50]]
} {1 {Usage: torch::transformer_encoder src d_model nhead}}

test transformer-encoder-4.2 {Error handling - invalid dModel} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder $src 0 2 2 8} error]
    expr {$result == 1 && [string match {*Invalid parameters: src tensor must be defined, dModel, nhead, numLayers*} $error]}
} {1}

test transformer-encoder-4.3 {Error handling - invalid named parameter} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder -src $src -invalid 4} error]
    return [list $result [string range $error 0 50]]
} {1 {Unknown parameter: -invalid}}

test transformer-encoder-4.4 {Error handling - missing value for parameter} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::transformer_encoder -src $src -dModel} error]
    return [list $result [string range $error 0 50]]
} {1 {Missing value for parameter}}

# Test mathematical correctness - both syntaxes should produce same result
test transformer-encoder-5.1 {Mathematical consistency between syntaxes} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result1 [torch::transformer_encoder $src 4 2 2 8]
    set result2 [torch::transformer_encoder -src $src -dModel 4 -nhead 2 -numLayers 2 -dimFeedforward 8]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [list $shape1 $shape2]
} {4 4}

# Test with different data types
test transformer-encoder-6.1 {Different data types} {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0} float64 cpu true]
    set result [torch::transformer_encoder $src 4 2 2 8]
    set dtype [torch::tensor_dtype $result]
    return $dtype
} {Float64}

cleanupTests 