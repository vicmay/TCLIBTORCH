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

# Helper function to create a test tensor
proc create_test_tensor {} {
    set data [list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
    set tensor [torch::tensor_create -data $data -dtype float32]
    ;# Reshape to 5D: batch_size=1, channels=1, depth=2, height=2, width=2
    set reshaped [torch::tensor_reshape $tensor {1 1 2 2 2}]
    return $reshaped
}

# Test cases for positional syntax
test reflection_pad3d-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set result [torch::reflection_pad3d $input {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 4}

test reflection_pad3d-1.2 {Positional syntax with uneven padding} {
    set input [create_test_tensor]
    set result [torch::reflection_pad3d $input {1 0 1 0 1 0}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 3 3 3}

# Test cases for named parameter syntax
test reflection_pad3d-2.1 {Basic named parameter syntax} {
    set input [create_test_tensor]
    set result [torch::reflectionPad3d -input $input -padding {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 4}

test reflection_pad3d-2.2 {Named parameter syntax with uneven padding} {
    set input [create_test_tensor]
    set result [torch::reflectionPad3d -input $input -padding {1 0 1 0 1 0}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 3 3 3}

test reflection_pad3d-2.3 {Named parameter syntax with -tensor alias} {
    set input [create_test_tensor]
    set result [torch::reflectionPad3d -tensor $input -padding {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 4}

test reflection_pad3d-2.4 {Named parameter syntax with -pad alias} {
    set input [create_test_tensor]
    set result [torch::reflectionPad3d -input $input -pad {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 4}

# Error handling tests
test reflection_pad3d-3.1 {Error on missing input} {
    catch {torch::reflectionPad3d -padding {1 1 1 1 1 1}} err
    set err
} {Required parameters missing: input tensor and padding values required}

test reflection_pad3d-3.2 {Error on missing padding} {
    set input [create_test_tensor]
    catch {torch::reflectionPad3d -input $input} err
    set err
} {Missing value for parameter}

test reflection_pad3d-3.3 {Error on invalid padding size} {
    set input [create_test_tensor]
    catch {torch::reflectionPad3d -input $input -padding {1 1 1}} err
    set err
} {Padding must be a list of 6 values for 3D}

test reflection_pad3d-3.4 {Error on invalid padding values} {
    set input [create_test_tensor]
    catch {torch::reflectionPad3d -input $input -padding {1 a 1 1 1 1}} err
    set err
} {Invalid padding value}

test reflection_pad3d-3.5 {Error on invalid tensor name} {
    catch {torch::reflectionPad3d -input invalid_tensor -padding {1 1 1 1 1 1}} err
    set err
} {Invalid tensor name}

test reflection_pad3d-3.6 {Error on wrong tensor dimensions} {
    set data [list 1.0 2.0 3.0 4.0]
    set tensor [torch::tensor_create -data $data -dtype float32]
    set reshaped [torch::tensor_reshape $tensor {2 2}]
    catch {torch::reflectionPad3d -input $reshaped -padding {1 1 1 1 1 1}} err
    set err
} {Expected 5D tensor for 3D padding, but got 2D tensor}

test reflection_pad3d-3.7 {Error on negative padding} {
    set input [create_test_tensor]
    catch {torch::reflectionPad3d -input $input -padding {-1 1 1 1 1 1}} err
    set err
} {Invalid padding value: padding cannot be negative}

cleanupTests 