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

# Basic conv3d tests
test conv3d-1.1 {Basic positional syntax - input and weight only} {
    # Create input tensor [batch, in_channels, depth, height, width]
    set input [torch::randn -shape {1 3 8 16 16}]
    
    # Create weight tensor [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    # Perform 3D convolution using positional syntax
    set result [torch::conv3d $input $weight]
    
    # Check that result tensor is created
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-1.2 {Basic positional syntax - with bias} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    set result [torch::conv3d $input $weight $bias]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-1.3 {Positional syntax - with bias "none"} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d $input $weight none]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-1.4 {Positional syntax - with stride as integer} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d $input $weight none 2]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-1.5 {Positional syntax - with stride as list} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d $input $weight none {2 1 2}]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-1.6 {Positional syntax - with padding as integer} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d $input $weight none 1 1]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-1.7 {Positional syntax - with padding as list} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d $input $weight none 1 {1 0 1}]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-1.8 {Positional syntax - with all parameters} {
    set input [torch::randn -shape {1 6 8 16 16}]
    set weight [torch::randn -shape {12 3 3 3 3}]
    set bias [torch::randn -shape {12}]
    
    # conv3d input weight bias stride padding dilation groups
    set result [torch::conv3d $input $weight $bias 1 1 1 2]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

# Named parameter syntax tests
test conv3d-2.1 {Named parameter syntax - basic} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d -input $input -weight $weight]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-2.2 {Named parameter syntax - with bias} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    set result [torch::conv3d -input $input -weight $weight -bias $bias]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-2.3 {Named parameter syntax - with stride as integer} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d -input $input -weight $weight -stride 2]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-2.4 {Named parameter syntax - with stride as list} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d -input $input -weight $weight -stride {2 1 2}]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-2.5 {Named parameter syntax - with all parameters} {
    set input [torch::randn -shape {1 6 8 16 16}]
    set weight [torch::randn -shape {12 3 3 3 3}]
    set bias [torch::randn -shape {12}]
    
    set result [torch::conv3d -input $input -weight $weight -bias $bias -stride {1 2 1} -padding {1 0 1} -dilation {1 1 1} -groups 2]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-2.6 {Named parameter syntax - parameter order variation} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    # Parameters in different order
    set result [torch::conv3d -stride 2 -weight $weight -input $input -padding 1]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

# CamelCase tests (conv3d is already camelCase)
test conv3d-3.1 {CamelCase command with positional syntax} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d $input $weight]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

test conv3d-3.2 {CamelCase command with named parameters} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [torch::conv3d -input $input -weight $weight]
    
    set shape [torch::tensor_shape $result]
    llength $shape
} {5}

# Error handling tests
test conv3d-4.1 {Error handling - missing required parameter in named syntax} {
    set input [torch::randn -shape {1 3 8 16 16}]
    
    set result [catch {torch::conv3d -input $input} error]
    
    list $result [string match "*Required parameters*" $error]
} {1 1}

test conv3d-4.2 {Error handling - invalid input tensor handle} {
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [catch {torch::conv3d -input "invalid_handle" -weight $weight} error]
    
    list $result [string match "*Invalid input tensor name*" $error]
} {1 1}

test conv3d-4.3 {Error handling - invalid weight tensor handle} {
    set input [torch::randn -shape {1 3 8 16 16}]
    
    set result [catch {torch::conv3d -input $input -weight "invalid_handle"} error]
    
    list $result [string match "*Invalid weight tensor name*" $error]
} {1 1}

test conv3d-4.4 {Error handling - unknown parameter} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [catch {torch::conv3d -input $input -weight $weight -unknown_param value} error]
    
    list $result [string match "*Unknown parameter*" $error]
} {1 1}

test conv3d-4.5 {Error handling - invalid stride list} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [catch {torch::conv3d -input $input -weight $weight -stride {1 2}} error]
    
    list $result [string match "*Value must be int or list of 3 ints*" $error]
} {1 1}

test conv3d-4.6 {Error handling - invalid padding list} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [catch {torch::conv3d -input $input -weight $weight -padding {1 2 3 4}} error]
    
    list $result [string match "*Value must be int or list of 3 ints*" $error]
} {1 1}

test conv3d-4.7 {Error handling - invalid groups value} {
    set input [torch::randn -shape {1 3 8 16 16}]
    set weight [torch::randn -shape {16 3 3 3 3}]
    
    set result [catch {torch::conv3d -input $input -weight $weight -groups "invalid"} error]
    
    list $result [string match "*Invalid groups value*" $error]
} {1 1}

# Integration tests - verify mathematical correctness
test conv3d-5.1 {Integration test - verify output dimensions} {
    # Input: [1, 2, 4, 8, 8] - batch=1, channels=2, depth=4, height=8, width=8
    set input [torch::ones -shape {1 2 4 8 8}]
    
    # Weight: [3, 2, 2, 3, 3] - out_channels=3, in_channels=2, kernel=(2,3,3)
    set weight [torch::ones -shape {3 2 2 3 3}]
    
    # Expected output with default stride=1, padding=0:
    # depth: 4 - 2 + 1 = 3
    # height: 8 - 3 + 1 = 6  
    # width: 8 - 3 + 1 = 6
    set result [torch::conv3d -input $input -weight $weight]
    
    set shape [torch::tensor_shape $result]
    
    # Should be [1, 3, 3, 6, 6]
    set expected_shape {1 3 3 6 6}
    expr {$shape eq $expected_shape}
} {1}

test conv3d-5.2 {Integration test - verify padding effect} {
    # Input: [1, 1, 4, 4, 4]
    set input [torch::ones -shape {1 1 4 4 4}]
    
    # Weight: [1, 1, 3, 3, 3] - 3x3x3 kernel
    set weight [torch::ones -shape {1 1 3 3 3}]
    
    # With padding=1, output should be same size as input
    set result [torch::conv3d -input $input -weight $weight -padding 1]
    
    set shape [torch::tensor_shape $result]
    
    # Should be [1, 1, 4, 4, 4]
    set expected_shape {1 1 4 4 4}
    expr {$shape eq $expected_shape}
} {1}

test conv3d-5.3 {Integration test - verify stride effect} {
    # Input: [1, 1, 8, 8, 8]
    set input [torch::ones -shape {1 1 8 8 8}]
    
    # Weight: [1, 1, 2, 2, 2] - 2x2x2 kernel
    set weight [torch::ones -shape {1 1 2 2 2}]
    
    # With stride=2, output should be half the size
    set result [torch::conv3d -input $input -weight $weight -stride 2]
    
    set shape [torch::tensor_shape $result]
    
    # Should be [1, 1, 4, 4, 4] - (8-2)/2+1 = 4 for each spatial dimension
    set expected_shape {1 1 4 4 4}
    expr {$shape eq $expected_shape}
} {1}

test conv3d-5.4 {Syntax consistency test - both syntaxes produce same result} {
    set input [torch::randn -shape {1 2 4 8 8}]
    set weight [torch::randn -shape {3 2 2 3 3}]
    set bias [torch::randn -shape {3}]
    
    # Positional syntax
    set result1 [torch::conv3d $input $weight $bias {1 2 1} {0 1 0} {1 1 1} 1]
    
    # Named parameter syntax
    set result2 [torch::conv3d -input $input -weight $weight -bias $bias -stride {1 2 1} -padding {0 1 0} -dilation {1 1 1} -groups 1]
    
    # Results should be equal (within floating point tolerance)
    set diff [torch::tensor_sub $result1 $result2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_diff_value [torch::tensor_item $max_diff]
    
    # Should be very close to 0
    expr {$max_diff_value < 1e-6}
} {1}

cleanupTests 