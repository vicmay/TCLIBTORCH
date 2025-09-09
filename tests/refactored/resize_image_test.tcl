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
    set data [list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0]
    set tensor [torch::tensor_create -data $data -dtype float32]
    ;# Reshape to 4D: batch_size=1, channels=1, height=3, width=4
    set reshaped [torch::tensor_reshape $tensor {1 1 3 4}]
    return $reshaped
}

# Test cases for positional syntax
test resize_image-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set result [torch::resize_image $input {6 8}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

test resize_image-1.2 {Positional syntax with mode} {
    set input [create_test_tensor]
    set result [torch::resize_image $input {6 8} nearest]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

test resize_image-1.3 {Positional syntax with mode and align_corners} {
    set input [create_test_tensor]
    set result [torch::resize_image $input {6 8} bilinear 1]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

# Test cases for named parameter syntax
test resize_image-2.1 {Basic named parameter syntax} {
    set input [create_test_tensor]
    set result [torch::resizeImage -input $input -size {6 8}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

test resize_image-2.2 {Named parameter syntax with mode} {
    set input [create_test_tensor]
    set result [torch::resizeImage -input $input -size {6 8} -mode nearest]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

test resize_image-2.3 {Named parameter syntax with mode and align_corners} {
    set input [create_test_tensor]
    set result [torch::resizeImage -input $input -size {6 8} -mode bilinear -alignCorners 1]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

test resize_image-2.4 {Named parameter syntax with -tensor alias} {
    set input [create_test_tensor]
    set result [torch::resizeImage -tensor $input -size {6 8}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

test resize_image-2.5 {Named parameter syntax with -image alias} {
    set input [create_test_tensor]
    set result [torch::resizeImage -image $input -size {6 8}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {4 1 1 6 8}

# Error handling tests
test resize_image-3.1 {Error on missing input} {
    catch {torch::resizeImage -size {6 8}} error
    set error
} {Required parameters missing: input tensor and size required}

test resize_image-3.2 {Error on missing size} {
    set input [create_test_tensor]
    catch {torch::resizeImage -input $input} error
    set error
} {Required parameters missing: input tensor and size required}

test resize_image-3.3 {Error on invalid size list} {
    set input [create_test_tensor]
    catch {torch::resizeImage -input $input -size {6}} error
    set error
} {It is expected output_size equals to 2, but got size 1}

test resize_image-3.4 {Error on invalid mode} {
    set input [create_test_tensor]
    catch {torch::resizeImage -input $input -size {6 8} -mode invalid} error
    set error
} {Invalid mode: invalid. Valid modes are: nearest, bilinear, bicubic}

test resize_image-3.5 {Error on invalid tensor name} {
    catch {torch::resizeImage -input invalid_tensor -size {6 8}} error
    set error
} {Invalid input tensor}

test resize_image-3.6 {Error on invalid align_corners} {
    set input [create_test_tensor]
    catch {torch::resizeImage -input $input -size {6 8} -alignCorners invalid} error
    set error
} {Invalid align_corners value}

test resize_image-3.7 {Error on unknown parameter} {
    set input [create_test_tensor]
    catch {torch::resizeImage -input $input -size {6 8} -unknown value} error
    set error
} {Unknown parameter: -unknown. Valid parameters are: -input/-tensor/-image, -size, -mode, -align_corners/-alignCorners}

test resize_image-3.8 {Error on align_corners with nearest mode} {
    set input [create_test_tensor]
    catch {torch::resizeImage -input $input -size {6 8} -mode nearest -alignCorners 1} error
    set error
} {align_corners option can only be used with bilinear or bicubic mode}

cleanupTests 