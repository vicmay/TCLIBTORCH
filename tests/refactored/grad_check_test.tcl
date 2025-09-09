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

# ============================================================================
# Test torch::grad_check - Gradient checking with dual syntax support
# ============================================================================

test grad_check-1.1 {Basic positional syntax} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check "dummy_func" $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-1.2 {Named parameter syntax} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-1.3 {Alternative named parameter syntax} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check -function "dummy_func" -input $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-1.4 {CamelCase alias} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::gradCheck "dummy_func" $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-1.5 {CamelCase alias with named parameters} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::gradCheck -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

# ============================================================================
# Error Handling Tests
# ============================================================================

test grad_check-2.1 {Missing arguments - positional} -body {
    torch::grad_check "dummy_func"
} -returnCodes error -match glob -result "*Usage: torch::grad_check func inputs*"

test grad_check-2.2 {Missing arguments - named} -body {
    torch::grad_check -func "dummy_func"
} -returnCodes error -match glob -result "*Required parameters missing*"

test grad_check-2.3 {Invalid tensor handle} -body {
    torch::grad_check -func "dummy_func" -inputs "invalid_tensor"
} -returnCodes error -match glob -result "*Invalid tensor handle*"

test grad_check-2.4 {Unknown parameter} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check -func "dummy_func" -inputs $input -unknown "value"
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -returnCodes error -match glob -result "*Unknown parameter*"

test grad_check-2.5 {Missing function parameter} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -returnCodes error -match glob -result "*Required parameters missing*"

test grad_check-2.6 {Missing inputs parameter} -body {
    torch::grad_check -func "dummy_func"
} -returnCodes error -match glob -result "*Required parameters missing*"

# ============================================================================
# Backward Compatibility Tests
# ============================================================================

test grad_check-3.1 {Backward compatibility - positional matches named} -setup {
    set input [torch::randn {2 3}]
} -body {
    set result1 [torch::grad_check "dummy_func" $input]
    set result2 [torch::grad_check -func "dummy_func" -inputs $input]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-3.2 {CamelCase alias produces same result} -setup {
    set input [torch::randn {2 3}]
} -body {
    set result1 [torch::grad_check "dummy_func" $input]
    set result2 [torch::gradCheck "dummy_func" $input]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

# ============================================================================
# Different tensor types and configurations
# ============================================================================

test grad_check-4.1 {Different tensor dtype} -setup {
    set input [torch::randn {3 2} cpu float64]
} -body {
    torch::grad_check -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-4.2 {Different tensor shape} -setup {
    set input [torch::randn {5 4 3}]
} -body {
    torch::grad_check -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-4.3 {1D tensor} -setup {
    set input [torch::randn {10}]
} -body {
    torch::grad_check -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check-4.4 {Scalar tensor} -setup {
    set input [torch::randn {}]
} -body {
    torch::grad_check -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

cleanupTests 