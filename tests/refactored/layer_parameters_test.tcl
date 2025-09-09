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
test layer_parameters-1.1 {Basic positional syntax} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 4 -outFeatures 2]
    
    ;# Test positional syntax
    set params [torch::layer_parameters $layer]
    
    ;# Should return a list of parameter tensor handles
    set param_count [llength $params]
    
    ;# Linear layer should have 2 parameters: weight and bias
    if {$param_count == 2} {
        return "success"
    } else {
        return "failed: expected 2 parameters, got $param_count"
    }
} -result "success"

;# Test cases for named parameter syntax
test layer_parameters-2.1 {Named parameter syntax} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 3 -outFeatures 1]
    
    ;# Test named parameter syntax
    set params [torch::layer_parameters -layer $layer]
    
    ;# Should return a list of parameter tensor handles
    set param_count [llength $params]
    
    ;# Linear layer should have 2 parameters: weight and bias
    if {$param_count == 2} {
        return "success"
    } else {
        return "failed: expected 2 parameters, got $param_count"
    }
} -result "success"

;# Test cases for camelCase alias
test layer_parameters-3.1 {camelCase alias - positional syntax} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    
    ;# Test camelCase alias with positional syntax
    set params [torch::layerParameters $layer]
    
    ;# Should return a list of parameter tensor handles
    set param_count [llength $params]
    
    ;# Linear layer should have 2 parameters: weight and bias
    if {$param_count == 2} {
        return "success"
    } else {
        return "failed: expected 2 parameters, got $param_count"
    }
} -result "success"

test layer_parameters-3.2 {camelCase alias - named parameter syntax} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 5 -outFeatures 3]
    
    ;# Test camelCase alias with named parameter syntax
    set params [torch::layerParameters -layer $layer]
    
    ;# Should return a list of parameter tensor handles
    set param_count [llength $params]
    
    ;# Linear layer should have 2 parameters: weight and bias
    if {$param_count == 2} {
        return "success"
    } else {
        return "failed: expected 2 parameters, got $param_count"
    }
} -result "success"

;# Test cases for parameter validation
test layer_parameters-4.1 {Parameter validation - both syntaxes produce same result} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 3 -outFeatures 2]
    
    ;# Test both syntaxes
    set params_positional [torch::layer_parameters $layer]
    set params_named [torch::layer_parameters -layer $layer]
    
    ;# Both should return the same parameter handles
    if {[llength $params_positional] == [llength $params_named]} {
        return "success"
    } else {
        return "failed: different parameter counts"
    }
} -result "success"

;# Error handling tests
test layer_parameters-5.1 {Error handling - invalid layer} -body {
    ;# Test with non-existent layer
    if {[catch {torch::layer_parameters "invalid_layer"} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test layer_parameters-5.2 {Error handling - missing required parameter} -body {
    ;# Test named syntax without required parameter
    if {[catch {torch::layer_parameters} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test layer_parameters-5.3 {Error handling - unknown parameter} -body {
    ;# Test named syntax with unknown parameter
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    if {[catch {torch::layer_parameters -unknown_param $layer} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

;# Test with different layer types
test layer_parameters-6.1 {Different layer types - conv2d} -body {
    ;# Create a conv2d layer
    set layer [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    ;# Test parameter extraction
    set params [torch::layer_parameters -layer $layer]
    
    ;# Conv2d layer should have 2 parameters: weight and bias
    set param_count [llength $params]
    if {$param_count == 2} {
        return "success"
    } else {
        return "failed: expected 2 parameters, got $param_count"
    }
} -result "success"

;# Test parameter tensor validity
test layer_parameters-7.1 {Parameter tensor validity} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    
    ;# Get parameters
    set params [torch::layer_parameters -layer $layer]
    
    ;# Check that each parameter is a valid tensor
    foreach param $params {
        if {[catch {torch::tensor_shape $param} shape]} {
            return "failed: invalid tensor handle $param"
        }
    }
    
    return "success"
} -result "success"

cleanupTests 