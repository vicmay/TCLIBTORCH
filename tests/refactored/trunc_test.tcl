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

# Helper to create a tensor with fractional values
proc create_fractional_tensor {} {
    return [torch::tensor_create {1.7 -2.3 3.8 -4.1 5.9} float32 cpu true]
}

# Test cases for positional syntax
test trunc-1.1 {Basic positional syntax} {
    set t [create_fractional_tensor]
    set trunc [torch::trunc $t]
    set vals [torch::tensor_to_list $trunc]
    return $vals
} {1.0 -2.0 3.0 -4.0 5.0}

# Test cases for named parameter syntax
test trunc-2.1 {Named parameter syntax} {
    set t [create_fractional_tensor]
    set trunc [torch::trunc -input $t]
    set vals [torch::tensor_to_list $trunc]
    return $vals
} {1.0 -2.0 3.0 -4.0 5.0}

test trunc-2.2 {Named parameter syntax with -tensor} {
    set t [create_fractional_tensor]
    set trunc [torch::trunc -tensor $t]
    set vals [torch::tensor_to_list $trunc]
    return $vals
} {1.0 -2.0 3.0 -4.0 5.0}

# Error handling tests
test trunc-3.1 {Error handling - missing input} {
    set result [catch {torch::trunc} error]
    return [list $result [string range $error 0 50]]
} {1 {Usage: torch::trunc tensor | torch::trunc -input te}}

test trunc-3.2 {Error handling - unknown named parameter} {
    set t [create_fractional_tensor]
    set result [catch {torch::trunc -foo $t} error]
    return $result
} {1}

test trunc-3.3 {Error handling - missing value for parameter} {
    set t [create_fractional_tensor]
    set result [catch {torch::trunc -input} error]
    return $result
} {1}

test trunc-3.4 {Error handling - invalid tensor name} {
    set result [catch {torch::trunc nonexistent_tensor} error]
    return $result
} {1}

# Mathematical consistency
test trunc-4.1 {Mathematical consistency between syntaxes} {
    set t [create_fractional_tensor]
    set trunc1 [torch::trunc $t]
    set trunc2 [torch::trunc -input $t]
    set vals1 [torch::tensor_to_list $trunc1]
    set vals2 [torch::tensor_to_list $trunc2]
    return [list $vals1 $vals2]
} {{1.0 -2.0 3.0 -4.0 5.0} {1.0 -2.0 3.0 -4.0 5.0}}

# Edge cases
test trunc-5.1 {Edge case - zero values} {
    set t [torch::tensor_create {0.0 -0.0 0.1 -0.1} float32 cpu true]
    set trunc [torch::trunc $t]
    set vals [torch::tensor_to_list $trunc]
    return $vals
} {0.0 -0.0 0.0 -0.0}

test trunc-5.2 {Edge case - large values} {
    set t [torch::tensor_create {123456.789 -987654.321} float32 cpu true]
    set trunc [torch::trunc $t]
    set vals [torch::tensor_to_list $trunc]
    return $vals
} {123456.0 -987654.0}

cleanupTests 