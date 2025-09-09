#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Helper function to create a state tensor
proc create_state_tensor {} {
    set state [torch::empty {64} -dtype int64]
    torch::fill_ $state 42
    return $state
}

;# Test cases for positional syntax
test set_rng_state-1.1 {Basic positional syntax} {
    set state [create_state_tensor]
    torch::set_rng_state $state
    torch::get_rng_state
} $state

;# Test cases for named syntax
test set_rng_state-2.1 {Named parameter syntax with -stateTensor} {
    set state [create_state_tensor]
    torch::set_rng_state -stateTensor $state
    torch::get_rng_state
} $state

test set_rng_state-2.2 {Named parameter syntax with -state_tensor} {
    set state [create_state_tensor]
    torch::set_rng_state -state_tensor $state
    torch::get_rng_state
} $state

;# Test cases for camelCase alias
test set_rng_state-3.1 {CamelCase alias} {
    set state [create_state_tensor]
    torch::setRngState -stateTensor $state
    torch::get_rng_state
} $state

;# Error handling tests
test set_rng_state-4.1 {Error handling - missing argument} -body {
    torch::set_rng_state
} -returnCodes error -result {Usage: torch::set_rng_state state_tensor | torch::set_rng_state -stateTensor tensor}

test set_rng_state-4.2 {Error handling - invalid tensor name} -body {
    torch::set_rng_state invalid_tensor
} -returnCodes error -result {Error in set_rng_state: Invalid tensor name: invalid_tensor}

test set_rng_state-4.3 {Error handling - too many arguments} -body {
    set state [create_state_tensor]
    torch::set_rng_state $state extra_arg
} -returnCodes error -result {Usage: torch::set_rng_state state_tensor}

;# Verify state affects random generation
test set_rng_state-5.1 {State affects random generation} {
    ;# First sequence
    set state [torch::get_rng_state]
    set rand1 [torch::rand {5}]
    
    ;# Reset state and generate again
    torch::set_rng_state $state
    set rand2 [torch::rand {5}]
    
    ;# Should be identical
    torch::equal $rand1 $rand2
} 1

cleanupTests 