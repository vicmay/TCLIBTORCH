#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    if {[file exists ../../build/libtorchtcl.so]} {
        puts "File exists but failed to load. Might be missing dependencies."
        puts "Try: LD_LIBRARY_PATH=../../libtorch/lib tclsh [info script]"
    } else {
        puts "File does not exist. Build the extension first."
    }
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create a test tensor
proc create_test_tensor {batch_size channels height width} {
    set tensor [torch::ones [list $batch_size $channels $height $width]]
    return $tensor
}

# Test cases for positional syntax
test conv2d-1.1 "Basic positional syntax" {
    set conv [torch::conv2d 3 16 3]
    expr {[string match "conv2d*" $conv]}
} {1}

test conv2d-1.2 "Forward pass with positional syntax" {
    set conv [torch::conv2d 3 16 3]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    # Output shape should be [batch_size, out_channels, height_out, width_out]
    # With default stride=1, padding=0, height_out = height_in - kernel_size + 1
    expr {$shape eq "2 16 30 30"}
} {1}

test conv2d-1.3 "Positional syntax with stride" {
    set conv [torch::conv2d 3 16 3 2]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    # With stride=2: height_out = (height_in - kernel_size) / stride + 1
    expr {$shape eq "2 16 15 15"}
} {1}

test conv2d-1.4 "Positional syntax with stride and padding" {
    set conv [torch::conv2d 3 16 3 1 1]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    # With padding=1: height_out = height_in - kernel_size + 2*padding + 1
    expr {$shape eq "2 16 32 32"}
} {1}

test conv2d-1.5 "Positional syntax with all parameters" {
    set conv [torch::conv2d 3 16 3 2 1 0]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "2 16 16 16"}
} {1}

# Test cases for named parameter syntax
test conv2d-2.1 "Basic named parameter syntax" {
    set conv [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]
    expr {[string match "conv2d*" $conv]}
} {1}

test conv2d-2.2 "Named parameter syntax with all parameters" {
    set conv [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -stride 2 -padding 1 -bias 1]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "2 16 16 16"}
} {1}

test conv2d-2.3 "Named parameter syntax with bias=false" {
    set conv [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -bias 0]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "2 16 30 30"}
} {1}

test conv2d-2.4 "Named parameter syntax with parameters in different order" {
    set conv [torch::conv2d -bias 1 -outChannels 16 -stride 1 -inChannels 3 -kernelSize 3 -padding 1]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "2 16 32 32"}
} {1}

# Test cases for camelCase alias
test conv2d-3.1 "Basic camelCase alias" {
    set conv [torch::conv2dLayer -inChannels 3 -outChannels 16 -kernelSize 3]
    expr {[string match "conv2d*" $conv]}
} {1}

test conv2d-3.2 "CamelCase alias with all parameters" {
    set conv [torch::conv2dLayer -inChannels 3 -outChannels 16 -kernelSize 3 -stride 2 -padding 1 -bias 1]
    set input [create_test_tensor 2 3 32 32]
    set output [torch::layer_forward $conv $input]
    set shape [torch::tensor_shape $output]
    expr {$shape eq "2 16 16 16"}
} {1}

# Error handling tests
test conv2d-4.1 "Error: Missing required parameters (named syntax)" {
    catch {torch::conv2d -inChannels 3 -outChannels 16} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test conv2d-4.2 "Error: Invalid inChannels parameter" {
    catch {torch::conv2d -inChannels -5 -outChannels 16 -kernelSize 3} result
    expr {[string match "*Required parameters missing or invalid*" $result]}
} {1}

test conv2d-4.3 "Error: Invalid kernelSize parameter" {
    catch {torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 0} result
    expr {[string match "*Required parameters missing or invalid*" $result]}
} {1}

test conv2d-4.4 "Error: Unknown parameter" {
    catch {torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -unknown 1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test conv2d-4.5 "Error: Missing value for parameter" {
    catch {torch::conv2d -inChannels 3 -outChannels 16 -kernelSize} result
    expr {[string match "*Missing value*" $result]}
} {1}

# Verify both syntaxes produce identical results
test conv2d-5.1 "Verify both syntaxes produce identical shapes" {
    set conv1 [torch::conv2d 3 16 3 2 1]
    set conv2 [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -stride 2 -padding 1]
    
    set input [create_test_tensor 2 3 32 32]
    
    set output1 [torch::layer_forward $conv1 $input]
    set output2 [torch::layer_forward $conv2 $input]
    
    expr {[torch::tensor_shape $output1] eq [torch::tensor_shape $output2]}
} {1}

cleanupTests 