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
    set data [list 1.0 2.0 3.0 4.0 5.0 6.0]
    set tensor [torch::tensor_create -data $data -dtype float32]
    ;# Reshape to 3D: batch_size=1, channels=1, width=6
    set reshaped [torch::tensor_reshape $tensor {1 1 6}]
    return $reshaped
}

# Test cases for positional syntax
test replication_pad1d-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set result [torch::replication_pad1d $input {1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {3 1 1 8}

test replication_pad1d-1.2 {Positional syntax with uneven padding} {
    set input [create_test_tensor]
    set result [torch::replication_pad1d $input {2 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {3 1 1 9}

# Test cases for named parameter syntax
test replication_pad1d-2.1 {Basic named parameter syntax} {
    set input [create_test_tensor]
    set result [torch::replicationPad1d -input $input -padding {1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {3 1 1 8}

test replication_pad1d-2.2 {Named parameter syntax with uneven padding} {
    set input [create_test_tensor]
    set result [torch::replicationPad1d -input $input -padding {2 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {3 1 1 9}

test replication_pad1d-2.3 {Named parameter syntax with -tensor alias} {
    set input [create_test_tensor]
    set result [torch::replicationPad1d -tensor $input -padding {1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {3 1 1 8}

test replication_pad1d-2.4 {Named parameter syntax with -pad alias} {
    set input [create_test_tensor]
    set result [torch::replicationPad1d -input $input -pad {1 1}]
    set shape [torch::tensor_shape $result]
    list [llength $shape] {*}$shape
} {3 1 1 8}

# Error handling tests
test replication_pad1d-3.1 {Error on missing input} {
    catch {torch::replicationPad1d -padding {1 1}} error
    set error
} {Required parameters missing: input tensor and padding values required}

test replication_pad1d-3.2 {Error on missing padding} {
    set input [create_test_tensor]
    catch {torch::replicationPad1d -input $input} error
    set error
} {Missing value for parameter}

test replication_pad1d-3.3 {Error on invalid padding format} {
    set input [create_test_tensor]
    catch {torch::replicationPad1d -input $input -padding {1}} error
    set error
} {Padding must be a list of 2 values for 1D}

test replication_pad1d-3.4 {Error on negative padding} {
    set input [create_test_tensor]
    catch {torch::replicationPad1d -input $input -padding {-1 1}} error
    set error
} {Invalid padding value: padding cannot be negative}

test replication_pad1d-3.5 {Error on invalid tensor dimensions} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    catch {torch::replicationPad1d -input $tensor -padding {1 1}} error
    set error
} {Expected 3D tensor (batch_size, channels, width) for 1D padding, but got 1D tensor}

test replication_pad1d-3.6 {Error on invalid tensor name} {
    catch {torch::replicationPad1d -input invalid_tensor -padding {1 1}} error
    set error
} {Invalid tensor name}

test replication_pad1d-3.7 {Error on unknown parameter} {
    set input [create_test_tensor]
    catch {torch::replicationPad1d -input $input -padding {1 1} -invalid value} error
    set error
} {Unknown parameter: -invalid. Valid parameters are: -input/-tensor, -padding/-pad}

cleanupTests 