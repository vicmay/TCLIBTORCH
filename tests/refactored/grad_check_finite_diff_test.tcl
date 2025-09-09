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
# Test torch::grad_check_finite_diff - Gradient checking with finite differences and dual syntax support
# ============================================================================

test grad_check_finite_diff-1.1 {Basic positional syntax} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff "dummy_func" $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-1.2 {Positional syntax with eps} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff "dummy_func" $input 1e-4
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-1.3 {Named parameter syntax} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-1.4 {Named parameter syntax with eps} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-6
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-1.5 {Alternative named parameter syntax} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -function "dummy_func" -input $input -epsilon 1e-7
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-1.6 {CamelCase alias} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::gradCheckFiniteDiff "dummy_func" $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-1.7 {CamelCase alias with named parameters} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::gradCheckFiniteDiff -func "dummy_func" -inputs $input -eps 1e-5
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

# ============================================================================
# Error Handling Tests
# ============================================================================

test grad_check_finite_diff-2.1 {Missing arguments - positional} -body {
    torch::grad_check_finite_diff "dummy_func"
} -returnCodes error -match glob -result "*Usage: torch::grad_check_finite_diff func inputs*"

test grad_check_finite_diff-2.2 {Missing arguments - named} -body {
    torch::grad_check_finite_diff -func "dummy_func"
} -returnCodes error -match glob -result "*Required parameters missing*"

test grad_check_finite_diff-2.3 {Invalid tensor handle} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs "invalid_tensor"
} -returnCodes error -match glob -result "*Invalid tensor handle*"

test grad_check_finite_diff-2.4 {Unknown parameter} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -unknown "value"
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -returnCodes error -match glob -result "*Unknown parameter*"

test grad_check_finite_diff-2.5 {Missing function parameter} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -returnCodes error -match glob -result "*Required parameters missing*"

test grad_check_finite_diff-2.6 {Invalid eps value} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps "invalid"
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -returnCodes error -match glob -result "*Invalid eps value*"

test grad_check_finite_diff-2.7 {Negative eps value} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps -1e-5
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -returnCodes error -match glob -result "*Required parameters missing*"

test grad_check_finite_diff-2.8 {Zero eps value} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 0.0
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -returnCodes error -match glob -result "*Required parameters missing*"

# ============================================================================
# Backward Compatibility Tests
# ============================================================================

test grad_check_finite_diff-3.1 {Backward compatibility - positional matches named} -setup {
    set input [torch::randn {2 3}]
} -body {
    set result1 [torch::grad_check_finite_diff "dummy_func" $input 1e-5]
    set result2 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-5]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-3.2 {CamelCase alias produces same result} -setup {
    set input [torch::randn {2 3}]
} -body {
    set result1 [torch::grad_check_finite_diff "dummy_func" $input]
    set result2 [torch::gradCheckFiniteDiff "dummy_func" $input]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-3.3 {Default eps handling} -setup {
    set input [torch::randn {2 3}]
} -body {
    # Both should work even without explicit eps
    set result1 [torch::grad_check_finite_diff "dummy_func" $input]
    set result2 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

# ============================================================================
# Different tensor types and configurations
# ============================================================================

test grad_check_finite_diff-4.1 {Different tensor dtype} -setup {
    set input [torch::randn {3 2} cpu float64]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-4.2 {Different tensor shape} -setup {
    set input [torch::randn {5 4 3}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-4
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-4.3 {1D tensor} -setup {
    set input [torch::randn {10}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-6
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-4.4 {Scalar tensor} -setup {
    set input [torch::randn {}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

# ============================================================================
# Epsilon value testing
# ============================================================================

test grad_check_finite_diff-5.1 {Very small eps value} -setup {
    set input [torch::randn {2 2}]
} -body {
    torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-10
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-5.2 {Larger eps value} -setup {
    set input [torch::randn {2 2}]
} -body {
    torch::grad_check_finite_diff "dummy_func" $input 1e-3
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-5.3 {Different eps values produce same result} -setup {
    set input [torch::randn {2 2}]
} -body {
    set result1 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-5]
    set result2 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-4]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

# ============================================================================
# Parameter order flexibility
# ============================================================================

test grad_check_finite_diff-6.1 {Parameter order flexibility - eps first} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -eps 1e-5 -func "dummy_func" -inputs $input
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-6.2 {Parameter order flexibility - inputs first} -setup {
    set input [torch::randn {2 3}]
} -body {
    torch::grad_check_finite_diff -inputs $input -func "dummy_func" -eps 1e-6
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

# ============================================================================
# Alias consistency tests
# ============================================================================

test grad_check_finite_diff-7.1 {Function parameter aliases} -setup {
    set input [torch::randn {2 2}]
} -body {
    set result1 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input]
    set result2 [torch::grad_check_finite_diff -function "dummy_func" -inputs $input]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-7.2 {Input parameter aliases} -setup {
    set input [torch::randn {2 2}]
} -body {
    set result1 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input]
    set result2 [torch::grad_check_finite_diff -func "dummy_func" -input $input]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

test grad_check_finite_diff-7.3 {Eps parameter aliases} -setup {
    set input [torch::randn {2 2}]
} -body {
    set result1 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input -eps 1e-5]
    set result2 [torch::grad_check_finite_diff -func "dummy_func" -inputs $input -epsilon 1e-5]
    expr {$result1 == $result2}
} -cleanup {
    # No cleanup needed - tensors are managed automatically
} -result {1}

cleanupTests 