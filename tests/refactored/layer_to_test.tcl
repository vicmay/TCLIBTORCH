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
test layer_to-1.1 {Basic positional syntax - move to CPU} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 4 -outFeatures 2]
    
    ;# Test positional syntax - move to CPU
    set result [torch::layer_to $layer cpu]
    
    ;# Should return the layer name
    if {$result eq $layer} {
        return "success"
    } else {
        return "failed: expected $layer, got $result"
    }
} -result "success"

test layer_to-1.2 {Positional syntax - verify device movement} -body {
    ;# Create a linear layer
    set layer [torch::linear -inFeatures 3 -outFeatures 1]
    
    ;# Move to CPU using positional syntax
    torch::layer_to $layer cpu
    
    ;# Verify the layer is on CPU
    set device [torch::layer_device $layer]
    
    if {$device eq "cpu"} {
        return "success"
    } else {
        return "failed: expected cpu, got $device"
    }
} -result "success"

;# Test cases for named parameter syntax
test layer_to-2.1 {Named parameter syntax - move to CPU} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    
    ;# Test named parameter syntax - move to CPU
    set result [torch::layer_to -layer $layer -device cpu]
    
    ;# Should return the layer name
    if {$result eq $layer} {
        return "success"
    } else {
        return "failed: expected $layer, got $result"
    }
} -result "success"

test layer_to-2.2 {Named parameter syntax - verify device movement} -body {
    ;# Create a linear layer
    set layer [torch::linear -inFeatures 5 -outFeatures 3]
    
    ;# Move to CPU using named syntax
    torch::layer_to -layer $layer -device cpu
    
    ;# Verify the layer is on CPU
    set device [torch::layer_device $layer]
    
    if {$device eq "cpu"} {
        return "success"
    } else {
        return "failed: expected cpu, got $device"
    }
} -result "success"

;# Test cases for camelCase alias
test layer_to-3.1 {camelCase alias - positional syntax} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 3 -outFeatures 2]
    
    ;# Test camelCase alias with positional syntax
    set result [torch::layerTo $layer cpu]
    
    ;# Should return the layer name
    if {$result eq $layer} {
        return "success"
    } else {
        return "failed: expected $layer, got $result"
    }
} -result "success"

test layer_to-3.2 {camelCase alias - named parameter syntax} -body {
    ;# Create a simple linear layer for testing
    set layer [torch::linear -inFeatures 4 -outFeatures 1]
    
    ;# Test camelCase alias with named parameter syntax
    set result [torch::layerTo -layer $layer -device cpu]
    
    ;# Should return the layer name
    if {$result eq $layer} {
        return "success"
    } else {
        return "failed: expected $layer, got $result"
    }
} -result "success"

;# Test cases for parameter validation
test layer_to-4.1 {Parameter validation - both syntaxes produce same result} -body {
    ;# Create two identical linear layers for testing
    set layer1 [torch::linear -inFeatures 3 -outFeatures 2]
    set layer2 [torch::linear -inFeatures 3 -outFeatures 2]
    
    ;# Test both syntaxes
    set result1 [torch::layer_to $layer1 cpu]
    set result2 [torch::layer_to -layer $layer2 -device cpu]
    
    ;# Both should return their respective layer names
    if {$result1 eq $layer1 && $result2 eq $layer2} {
        return "success"
    } else {
        return "failed: inconsistent results"
    }
} -result "success"

;# Error handling tests
test layer_to-5.1 {Error handling - invalid layer} -body {
    ;# Test with non-existent layer
    if {[catch {torch::layer_to "invalid_layer" cpu} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test layer_to-5.2 {Error handling - missing required parameters} -body {
    ;# Test named syntax without required parameters
    if {[catch {torch::layer_to} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test layer_to-5.3 {Error handling - missing device parameter} -body {
    ;# Test positional syntax with missing device
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    if {[catch {torch::layer_to $layer} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test layer_to-5.4 {Error handling - unknown parameter} -body {
    ;# Test named syntax with unknown parameter
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    if {[catch {torch::layer_to -unknown_param $layer -device cpu} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

;# Test with different layer types
test layer_to-6.1 {Different layer types - conv2d} -body {
    ;# Create a conv2d layer
    set layer [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    ;# Test device movement
    set result [torch::layer_to -layer $layer -device cpu]
    
    ;# Should return the layer name
    if {$result eq $layer} {
        return "success"
    } else {
        return "failed: expected $layer, got $result"
    }
} -result "success"

test layer_to-6.2 {Different layer types - sequential} -body {
    ;# Create a linear layer first
    set linear [torch::linear -inFeatures 10 -outFeatures 5]
    
    ;# Create a sequential layer with the linear layer
    set seq [torch::sequential [list $linear]]
    
    ;# Test device movement
    set result [torch::layer_to -layer $seq -device cpu]
    
    ;# Should return the layer name
    if {$result eq $seq} {
        return "success"
    } else {
        return "failed: expected $seq, got $result"
    }
} -result "success"

;# Test parameter order flexibility
test layer_to-7.1 {Parameter order flexibility} -body {
    ;# Create a linear layer
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    
    ;# Test different parameter orders
    set result1 [torch::layer_to -layer $layer -device cpu]
    set result2 [torch::layer_to -device cpu -layer $layer]
    
    ;# Both should work and return the layer name
    if {$result1 eq $layer && $result2 eq $layer} {
        return "success"
    } else {
        return "failed: parameter order should be flexible"
    }
} -result "success"

;# Test chaining operations
test layer_to-8.1 {Chaining operations} -body {
    ;# Create a linear layer
    set layer [torch::linear -inFeatures 4 -outFeatures 2]
    
    ;# Test chaining: move to CPU, then get device
    set moved_layer [torch::layer_to -layer $layer -device cpu]
    set device [torch::layer_device $moved_layer]
    
    if {$device eq "cpu"} {
        return "success"
    } else {
        return "failed: chaining operations failed"
    }
} -result "success"

;# Test device string validation
test layer_to-9.1 {Device string validation - cpu} -body {
    ;# Create a linear layer
    set layer [torch::linear -inFeatures 2 -outFeatures 1]
    
    ;# Test with various CPU device strings
    set result [torch::layer_to -layer $layer -device cpu]
    
    if {$result eq $layer} {
        return "success"
    } else {
        return "failed: CPU device string not accepted"
    }
} -result "success"

cleanupTests 