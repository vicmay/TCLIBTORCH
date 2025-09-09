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
    set data [list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0]
    set tensor [torch::tensor_create -data $data -dtype float32]
    ;# Reshape to 5D: batch_size=1, channels=1, depth=2, height=2, width=4
    set reshaped [torch::tensor_reshape $tensor {1 1 2 2 4}]
    return $reshaped
}

# Test cases for positional syntax
test replication_pad3d-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set result [torch::replication_pad3d $input {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 6}

test replication_pad3d-1.2 {Positional syntax with uneven padding} {
    set input [create_test_tensor]
    set result [torch::replication_pad3d $input {2 1 2 1 2 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 5 5 7}

# Test cases for named parameter syntax
test replication_pad3d-2.1 {Basic named parameter syntax} {
    set input [create_test_tensor]
    set result [torch::replicationPad3d -input $input -padding {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 6}

test replication_pad3d-2.2 {Named parameter syntax with uneven padding} {
    set input [create_test_tensor]
    set result [torch::replicationPad3d -input $input -padding {2 1 2 1 2 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 5 5 7}

test replication_pad3d-2.3 {Named parameter syntax with -tensor alias} {
    set input [create_test_tensor]
    set result [torch::replicationPad3d -tensor $input -padding {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 6}

test replication_pad3d-2.4 {Named parameter syntax with -pad alias} {
    set input [create_test_tensor]
    set result [torch::replicationPad3d -input $input -pad {1 1 1 1 1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {5 1 1 4 4 6}

# Error handling tests
test replication_pad3d-3.1 {Error on missing input} {
    catch {torch::replicationPad3d -padding {1 1 1 1 1 1}} error
    set error
} {Required parameters missing: input tensor and padding values required}

test replication_pad3d-3.2 {Error on missing padding} {
    set input [create_test_tensor]
    catch {torch::replicationPad3d -input $input} error
    set error
} {Missing value for parameter}

test replication_pad3d-3.3 {Error on invalid padding list size} {
    set input [create_test_tensor]
    catch {torch::replicationPad3d -input $input -padding {1 1 1 1 1}} error
    set error
} {Padding must be a list of 6 values for 3D}

test replication_pad3d-3.4 {Error on negative padding} {
    set input [create_test_tensor]
    catch {torch::replicationPad3d -input $input -padding {1 -1 1 1 1 1}} error
    set error
} {Invalid padding value: padding cannot be negative}

test replication_pad3d-3.5 {Error on invalid tensor name} {
    catch {torch::replicationPad3d -input invalid_tensor -padding {1 1 1 1 1 1}} error
    set error
} {Invalid tensor name}

test replication_pad3d-3.6 {Error on wrong tensor dimensions} {
    set data [list 1.0 2.0 3.0 4.0]
    set tensor [torch::tensor_create -data $data -dtype float32]
    set reshaped [torch::tensor_reshape $tensor {2 2}]
    catch {torch::replicationPad3d -input $reshaped -padding {1 1 1 1 1 1}} error
    set error
} {Expected 5D tensor for 3D padding, but got 2D tensor}

test replication_pad3d-3.7 {Error on unknown parameter} {
    set input [create_test_tensor]
    catch {torch::replicationPad3d -input $input -padding {1 1 1 1 1 1} -unknown value} error
    set error
} {Unknown parameter: -unknown. Valid parameters are: -input/-tensor, -padding/-pad}

cleanupTests 