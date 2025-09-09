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

;# Helper function to create test tensors
proc create_test_tensors {} {
    set tgt [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set memory [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2} -dtype float32]
    return [list $tgt $memory]
}

;# Helper function to cleanup tensors
proc cleanup_tensors {tensors} {
    foreach tensor $tensors {
        if {[info exists tensor] && $tensor ne ""} {
            catch {torch::tensor_delete $tensor}
        }
    }
}

;# Test cases for positional syntax
test transformer-decoder-1.1 {Basic positional syntax} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformer_decoder $tgt $memory 2 1 2 4]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 2}

test transformer-decoder-1.2 {Positional syntax with different parameters} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformer_decoder $tgt $memory 4 2 1 8]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 4}

;# Test cases for named parameter syntax
test transformer-decoder-2.1 {Named parameter syntax} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformer_decoder -tgt $tgt -memory $memory -dModel 2 -nhead 1 -numLayers 2 -dimFeedforward 4]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 2}

test transformer-decoder-2.2 {Named parameters with different values} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformer_decoder -tgt $tgt -memory $memory -dModel 4 -nhead 2 -numLayers 1 -dimFeedforward 8]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 4}

;# Test cases for camelCase alias
test transformer-decoder-3.1 {CamelCase alias} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformerDecoder $tgt $memory 2 1 2 4]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 2}

test transformer-decoder-3.2 {CamelCase alias with named parameters} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformerDecoder -tgt $tgt -memory $memory -dModel 2 -nhead 1 -numLayers 2 -dimFeedforward 4]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 2}

;# Error handling tests
test transformer-decoder-4.1 {Missing parameters - positional} {
    lassign [create_test_tensors] tgt memory
    set result [catch {torch::transformer_decoder $tgt $memory 2 1 2} msg]
    cleanup_tensors [list $tgt $memory]
    return [list $result $msg]
} {1 {Usage: torch::transformer_decoder tgt memory d_model nhead num_layers dim_feedforward}}

test transformer-decoder-4.2 {Invalid dModel value} {
    lassign [create_test_tensors] tgt memory
    set result [catch {torch::transformer_decoder $tgt $memory 0 1 2 4} msg]
    cleanup_tensors [list $tgt $memory]
    return [list $result $msg]
} {1 {Invalid parameters: all tensors must be defined, dModel, nhead, numLayers, and dimFeedforward must be positive}}

test transformer-decoder-4.3 {Invalid nhead value} {
    lassign [create_test_tensors] tgt memory
    set result [catch {torch::transformer_decoder $tgt $memory 2 -1 2 4} msg]
    cleanup_tensors [list $tgt $memory]
    return [list $result $msg]
} {1 {Invalid parameters: all tensors must be defined, dModel, nhead, numLayers, and dimFeedforward must be positive}}

test transformer-decoder-4.4 {Invalid numLayers value} {
    lassign [create_test_tensors] tgt memory
    set result [catch {torch::transformer_decoder $tgt $memory 2 1 0 4} msg]
    cleanup_tensors [list $tgt $memory]
    return [list $result $msg]
} {1 {Invalid parameters: all tensors must be defined, dModel, nhead, numLayers, and dimFeedforward must be positive}}

test transformer-decoder-4.5 {Invalid dimFeedforward value} {
    lassign [create_test_tensors] tgt memory
    set result [catch {torch::transformer_decoder $tgt $memory 2 1 2 -1} msg]
    cleanup_tensors [list $tgt $memory]
    return [list $result $msg]
} {1 {Invalid parameters: all tensors must be defined, dModel, nhead, numLayers, and dimFeedforward must be positive}}

test transformer-decoder-4.6 {Unknown named parameter} {
    lassign [create_test_tensors] tgt memory
    set result [catch {torch::transformer_decoder -tgt $tgt -memory $memory -dModel 2 -nhead 1 -numLayers 2 -dimFeedforward 4 -unknown 5} msg]
    cleanup_tensors [list $tgt $memory]
    return [list $result $msg]
} {1 {Unknown parameter: -unknown}}

test transformer-decoder-4.7 {Missing value for parameter} {
    lassign [create_test_tensors] tgt memory
    set result [catch {torch::transformer_decoder -tgt $tgt -memory $memory -dModel 2 -nhead} msg]
    cleanup_tensors [list $tgt $memory]
    return [list $result $msg]
} {1 {Missing value for parameter}}

;# Edge cases
test transformer-decoder-5.1 {Single layer transformer} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformer_decoder $tgt $memory 2 1 1 4]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 2}

test transformer-decoder-5.2 {Multiple layers transformer} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformer_decoder $tgt $memory 2 1 5 4]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 2}

test transformer-decoder-5.3 {Multiple attention heads} {
    lassign [create_test_tensors] tgt memory
    set result [torch::transformer_decoder $tgt $memory 4 2 2 8]
    set shape [torch::tensor_shape $result]
    cleanup_tensors [list $tgt $memory $result]
    return $shape
} {2 4}

;# Consistency tests - both syntaxes should produce same results
test transformer-decoder-6.1 {Consistency between positional and named syntax} {
    lassign [create_test_tensors] tgt memory
    set result1 [torch::transformer_decoder $tgt $memory 2 1 2 4]
    set result2 [torch::transformer_decoder -tgt $tgt -memory $memory -dModel 2 -nhead 1 -numLayers 2 -dimFeedforward 4]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    cleanup_tensors [list $tgt $memory $result1 $result2]
    return [list $shape1 $shape2]
} {{2 2} {2 2}}

test transformer-decoder-6.2 {Consistency between snake_case and camelCase} {
    lassign [create_test_tensors] tgt memory
    set result1 [torch::transformer_decoder $tgt $memory 2 1 2 4]
    set result2 [torch::transformerDecoder $tgt $memory 2 1 2 4]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    cleanup_tensors [list $tgt $memory $result1 $result2]
    return [list $shape1 $shape2]
} {{2 2} {2 2}}

cleanupTests 