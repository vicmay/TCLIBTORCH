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

# Basic conv2d_set_weights tests
test conv2d_set_weights-1.1 {Basic positional syntax - weight only} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight tensor
    set weight [torch::randn -shape {16 3 3 3}]
    
    # Set weights using positional syntax
    set result [torch::conv2d_set_weights $conv2d $weight]
    
    set result
} {OK}

test conv2d_set_weights-1.2 {Basic positional syntax - weight and bias} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight and bias tensors
    set weight [torch::randn -shape {16 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    # Set weights using positional syntax
    set result [torch::conv2d_set_weights $conv2d $weight $bias]
    
    set result  
} {OK}

test conv2d_set_weights-2.1 {Named parameter syntax - weight only} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight tensor
    set weight [torch::randn -shape {16 3 3 3}]
    
    # Set weights using named parameter syntax
    set result [torch::conv2d_set_weights -layer $conv2d -weight $weight]
    
    set result
} {OK}

test conv2d_set_weights-2.2 {Named parameter syntax - weight and bias} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight and bias tensors
    set weight [torch::randn -shape {16 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    # Set weights using named parameter syntax
    set result [torch::conv2d_set_weights -layer $conv2d -weight $weight -bias $bias]
    
    set result
} {OK}

test conv2d_set_weights-2.3 {Named parameter syntax - parameter order variation} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight and bias tensors
    set weight [torch::randn -shape {16 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    # Set weights using named parameter syntax with different order
    set result [torch::conv2d_set_weights -weight $weight -bias $bias -layer $conv2d]
    
    set result
} {OK}

test conv2d_set_weights-3.1 {CamelCase alias - weight only} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight tensor
    set weight [torch::randn -shape {16 3 3 3}]
    
    # Set weights using camelCase alias with named parameters
    set result [torch::conv2dSetWeights -layer $conv2d -weight $weight]
    
    set result
} {OK}

test conv2d_set_weights-3.2 {CamelCase alias - weight and bias} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight and bias tensors
    set weight [torch::randn -shape {16 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    # Set weights using camelCase alias with named parameters
    set result [torch::conv2dSetWeights -layer $conv2d -weight $weight -bias $bias]
    
    set result
} {OK}

test conv2d_set_weights-3.3 {CamelCase alias with positional syntax} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Create weight and bias tensors
    set weight [torch::randn -shape {16 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    # Set weights using camelCase alias with positional syntax (backward compatibility)
    set result [torch::conv2dSetWeights $conv2d $weight $bias]
    
    set result
} {OK}

# Error handling tests
test conv2d_set_weights-4.1 {Error handling - missing required parameter in named syntax} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Try to set weights without specifying weight parameter
    set result [catch {torch::conv2d_set_weights -layer $conv2d} error]
    
    list $result [string match "*Required parameters*" $error]
} {1 1}

test conv2d_set_weights-4.2 {Error handling - invalid layer handle} {
    # Create weight tensor
    set weight [torch::randn -shape {16 3 3 3}]
    
    # Try with invalid layer handle
    set result [catch {torch::conv2d_set_weights -layer "invalid_handle" -weight $weight} error]
    
    list $result [string match "*Invalid layer name*" $error]
} {1 1}

test conv2d_set_weights-4.3 {Error handling - invalid weight tensor handle} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    
    # Try with invalid weight tensor handle
    set result [catch {torch::conv2d_set_weights -layer $conv2d -weight "invalid_handle"} error]
    
    list $result [string match "*Invalid weight tensor name*" $error]
} {1 1}

test conv2d_set_weights-4.4 {Error handling - unknown parameter} {
    # Create a Conv2d layer and weight tensor
    set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    set weight [torch::randn -shape {16 3 3 3}]
    
    # Try with unknown parameter
    set result [catch {torch::conv2d_set_weights -layer $conv2d -weight $weight -unknown_param value} error]
    
    list $result [string match "*Unknown parameter*" $error]
} {1 1}

# Integration test - verify weights are actually set
test conv2d_set_weights-5.1 {Integration test - verify weight setting works} {
    # Create a Conv2d layer
    set conv2d [torch::conv2d -inChannels 1 -outChannels 1 -kernelSize 3 -bias false]
    
    # Create specific weight tensor (all ones)
    set weight [torch::ones -shape {1 1 3 3}]
    
    # Set weights
    torch::conv2d_set_weights -layer $conv2d -weight $weight
    
    # Create input tensor
    set input [torch::ones -shape {1 1 5 5}]
    
    # Forward pass
    set output [torch::layer_forward $conv2d $input]
    
    # Get output data - should be 9.0 at center position (3x3 filter of ones applied to input of ones)
    # Use mean to get a single value we can extract
    set output_mean [torch::tensor_mean $output]
    set first_value [torch::tensor_item $output_mean]
    
    # Should be 9.0 (3x3 = 9 ones multiplied)
    expr {$first_value == 9.0}
} {1}

cleanupTests 