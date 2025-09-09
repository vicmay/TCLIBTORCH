#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax
test adaptive_avgpool3d-1.1 {Basic positional syntax} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}
    set tensor [torch::tensor_create $data float32 cpu false]
    # Create 3D tensor
    set reshaped [torch::tensor_reshape $tensor {1 1 2 2 2}]
    set result [torch::adaptive_avgpool3d $reshaped 1]
    expr {[string length $result] > 0}
} {1}

# Test 2: Named parameter syntax  
test adaptive_avgpool3d-2.1 {Named parameter syntax} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}
    set tensor [torch::tensor_create $data float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {1 1 2 2 2}]
    set result [torch::adaptive_avgpool3d -input $reshaped -output_size {1 1 1}]
    expr {[string length $result] > 0}
} {1}

# Test 3: camelCase alias
test adaptive_avgpool3d-3.1 {camelCase alias} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}
    set tensor [torch::tensor_create $data float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {1 1 2 2 2}]
    set result [torch::adaptiveAvgpool3d -input $reshaped -output_size 1]
    expr {[string length $result] > 0}
} {1}

# Test 4: Error handling
test adaptive_avgpool3d-4.1 {Error handling} {
    set result [catch {torch::adaptive_avgpool3d -output_size 1} error]
    expr {$result == 1}
} {1}

cleanupTests 