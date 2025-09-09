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

# Helper function to create test tensors
proc create_test_tensor {} {
    return [torch::tensor_create {0 1 2 3 4 5 6 7 8 9 10 11} {3 4} float32]
}

# ============================================================================
# Basic Roll Tests - Positional Syntax
# ============================================================================

test roll-1.1 {Basic roll with single shift} {
    set input [create_test_tensor]
    set result [torch::roll $input {2}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-1.2 {Roll with multiple shifts and dims} {
    set input [create_test_tensor]
    set result [torch::roll $input {2 1} {0 1}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-1.3 {Roll with single shift and dim} {
    set input [create_test_tensor]
    set result [torch::roll $input {2} {0}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-1.4 {Roll with single shift and dim} {
    set input [create_test_tensor]
    set result [torch::roll $input {2} {1}]
    expr {[string match "tensor*" $result]}
} {1}

# ============================================================================
# Basic Roll Tests - Named Parameter Syntax
# ============================================================================

test roll-2.1 {Named parameter syntax - basic roll} {
    set input [create_test_tensor]
    set result [torch::roll -input $input -shifts {2}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-2.2 {Named parameter syntax - with dims} {
    set input [create_test_tensor]
    set result [torch::roll -input $input -shifts {2} -dims {0}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-2.3 {Named parameter syntax - multiple shifts and dims} {
    set input [create_test_tensor]
    set result [torch::roll -input $input -shifts {2 1} -dims {0 1}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-2.4 {Named parameter syntax - parameter order} {
    set input [create_test_tensor]
    set result [torch::roll -shifts {2} -input $input -dims {0}]
    expr {[string match "tensor*" $result]}
} {1}

# ============================================================================
# Error Handling Tests
# ============================================================================

test roll-3.1 {Error - missing input tensor} {
    catch {torch::roll} msg
    string match "*Usage:*" $msg
} {1}

test roll-3.2 {Error - missing shifts} {
    set input [create_test_tensor]
    catch {torch::roll $input} msg
    string match "*Usage:*" $msg
} {1}

test roll-3.3 {Error - invalid input tensor} {
    catch {torch::roll "invalid" {1}} msg
    string match "*Invalid input tensor*" $msg
} {1}

test roll-3.4 {Error - invalid shifts} {
    set input [create_test_tensor]
    set code [catch {torch::roll $input "not_a_list"} msg]
    expr {$code == 1}
} {1}

test roll-3.5 {Error - invalid dims} {
    set input [create_test_tensor]
    set code [catch {torch::roll $input {1} "not_a_list"} msg]
    expr {$code == 1}
} {1}

test roll-3.6 {Error - shifts/dims mismatch} {
    set input [create_test_tensor]
    set code [catch {torch::roll $input {1 2} {0}} msg]
    expr {$code == 1}
} {1}

# ============================================================================
# Edge Cases
# ============================================================================

test roll-4.1 {Edge case - zero shift} {
    set input [create_test_tensor]
    set result [torch::roll $input {0}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-4.2 {Edge case - negative shift} {
    set input [create_test_tensor]
    set result [torch::roll $input {-1}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-4.3 {Edge case - large shift} {
    set input [create_test_tensor]
    set result [torch::roll $input {100}]
    expr {[string match "tensor*" $result]}
} {1}

# ============================================================================
# CamelCase Alias Tests
# ============================================================================

test roll-5.1 {CamelCase alias - basic functionality} {
    set input [create_test_tensor]
    set result [torch::roll $input {2}]
    expr {[string match "tensor*" $result]}
} {1}

test roll-5.2 {CamelCase alias - with named parameters} {
    set input [create_test_tensor]
    set result [torch::roll -input $input -shifts {2} -dims {0}]
    expr {[string match "tensor*" $result]}
} {1}

cleanupTests 