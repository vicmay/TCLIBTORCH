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

# Helper function to create test tensors
proc create_test_tensors {} {
    set input [torch::tensor_create {1.0 2.0 3.0} float32]
    set other [torch::tensor_create {4.0 5.0} float32]
    return [list $input $other]
}

# Helper function to verify outer product result
proc verify_outer_product {result} {
    # Expected result should be a 3x2 matrix:
    # [[ 4.0  5.0]
    #  [ 8.0 10.0]
    #  [12.0 15.0]]
    set values [torch::tensor_to_list $result]
    set expected {4.0 5.0 8.0 10.0 12.0 15.0}
    
    # Verify each value with tolerance for floating point
    foreach v $values e $expected {
        if {abs($v - $e) > 1e-6} {
            return 0
        }
    }
    return 1
}

# Test cases for positional syntax
test outer-1.1 {Basic positional syntax} {
    lassign [create_test_tensors] input other
    set result [torch::outer $input $other]
    verify_outer_product $result
} {1}

# Test cases for named parameter syntax
test outer-2.1 {Named parameter syntax} {
    lassign [create_test_tensors] input other
    set result [torch::outer -input $input -other $other]
    verify_outer_product $result
} {1}

# Test cases for camelCase alias
test outer-3.1 {CamelCase alias with positional syntax} {
    lassign [create_test_tensors] input other
    set result [torch::Outer $input $other]
    verify_outer_product $result
} {1}

test outer-3.2 {CamelCase alias with named parameters} {
    lassign [create_test_tensors] input other
    set result [torch::Outer -input $input -other $other]
    verify_outer_product $result
} {1}

# Error handling tests
test outer-4.1 {Missing required parameters} {
    lassign [create_test_tensors] input other
    catch {torch::outer} result
    set result
} {Usage: torch::outer input other | torch::outer -input tensor -other tensor}

test outer-4.2 {Invalid input tensor} {
    lassign [create_test_tensors] input other
    catch {torch::outer invalid_tensor $other} result
    set result
} {Invalid input tensor}

test outer-4.3 {Invalid other tensor} {
    lassign [create_test_tensors] input other
    catch {torch::outer $input invalid_tensor} result
    set result
} {Invalid other tensor}

test outer-4.4 {Invalid parameter name} {
    lassign [create_test_tensors] input other
    catch {torch::outer -invalid $input -other $other} result
    set result
} {Unknown parameter: -invalid}

cleanupTests 