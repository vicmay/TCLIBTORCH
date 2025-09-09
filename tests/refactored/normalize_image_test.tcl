#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create a tensor
proc create_test_tensor {} {
    set data {1.0 2.0 3.0 4.0}
    set tensor [torch::tensor_create $data float32]
    return $tensor
}

# Helper function to create mean and std tensors
proc create_mean_std {} {
    set mean [torch::tensor_create {2.0} float32]
    set std [torch::tensor_create {2.0} float32]
    return [list $mean $std]
}

# Test cases for positional syntax
test normalize-1.1 {Basic positional syntax} {
    set tensor [create_test_tensor]
    lassign [create_mean_std] mean std
    set result [torch::normalize_image $tensor $mean $std]
    expr {[string length $result] > 0}
} {1}

test normalize-1.2 {Positional syntax with inplace} {
    set tensor [create_test_tensor]
    lassign [create_mean_std] mean std
    set result [torch::normalize_image $tensor $mean $std 1]
    expr {$result eq $tensor}
} {1}

# Test cases for named parameter syntax
test normalize-2.1 {Named parameter syntax} {
    set tensor [create_test_tensor]
    lassign [create_mean_std] mean std
    set result [torch::normalize_image -image $tensor -mean $mean -std $std]
    expr {[string length $result] > 0}
} {1}

test normalize-2.2 {Named parameter syntax with inplace} {
    set tensor [create_test_tensor]
    lassign [create_mean_std] mean std
    set result [torch::normalize_image -image $tensor -mean $mean -std $std -inplace 1]
    expr {$result eq $tensor}
} {1}

# Test cases for camelCase alias
test normalize-3.1 {CamelCase alias basic} {
    set tensor [create_test_tensor]
    lassign [create_mean_std] mean std
    set result [torch::normalizeImage -image $tensor -mean $mean -std $std]
    expr {[string length $result] > 0}
} {1}

test normalize-3.2 {CamelCase alias with inplace} {
    set tensor [create_test_tensor]
    lassign [create_mean_std] mean std
    set result [torch::normalizeImage -image $tensor -mean $mean -std $std -inplace 1]
    expr {$result eq $tensor}
} {1}

# Error handling tests
test normalize-4.1 {Missing required parameters} {
    set tensor [create_test_tensor]
    catch {torch::normalize_image $tensor} result
    set result
} {Error in normalize_image: Usage: torch::normalize_image image mean std ?inplace? | torch::normalize_image -image tensor -mean tensor -std tensor ?-inplace bool?}

test normalize-4.2 {Invalid tensor} {
    catch {torch::normalize_image invalid_tensor [lindex [create_mean_std] 0] [lindex [create_mean_std] 1]} result
    set result
} {Error in normalize_image: Invalid image tensor}

test normalize-4.3 {Invalid mean tensor} {
    set tensor [create_test_tensor]
    catch {torch::normalize_image $tensor invalid_mean [lindex [create_mean_std] 1]} result
    set result
} {Error in normalize_image: Invalid mean tensor}

test normalize-4.4 {Invalid std tensor} {
    set tensor [create_test_tensor]
    lassign [create_mean_std] mean std
    catch {torch::normalize_image $tensor $mean invalid_std} result
    set result
} {Error in normalize_image: Invalid std tensor}

cleanupTests 