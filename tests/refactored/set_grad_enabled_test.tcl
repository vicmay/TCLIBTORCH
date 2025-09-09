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

;# Test cases for positional syntax
test set_grad_enabled-1.1 {Basic positional syntax - enable} {
    torch::set_grad_enabled 1
    torch::is_grad_enabled
} {1}

test set_grad_enabled-1.2 {Basic positional syntax - disable} {
    torch::set_grad_enabled 0
    torch::is_grad_enabled
} {0}

;# Test cases for named syntax
test set_grad_enabled-2.1 {Named parameter syntax - enable} {
    torch::set_grad_enabled -enabled 1
    torch::is_grad_enabled
} {1}

test set_grad_enabled-2.2 {Named parameter syntax - disable} {
    torch::set_grad_enabled -enabled 0
    torch::is_grad_enabled
} {0}

;# Test cases for camelCase alias
test set_grad_enabled-3.1 {CamelCase alias - enable} {
    torch::setGradEnabled -enabled 1
    torch::is_grad_enabled
} {1}

test set_grad_enabled-3.2 {CamelCase alias - disable} {
    torch::setGradEnabled -enabled 0
    torch::is_grad_enabled
} {0}

;# Error handling tests
test set_grad_enabled-4.1 {Error handling - missing argument} -body {
    torch::set_grad_enabled
} -returnCodes error -result {Error in set_grad_enabled: Usage: torch::set_grad_enabled enabled | torch::set_grad_enabled -enabled value}

test set_grad_enabled-4.2 {Error handling - invalid argument type} -body {
    torch::set_grad_enabled invalid
} -returnCodes error -result {Error in set_grad_enabled: Invalid enabled value (must be boolean)}

test set_grad_enabled-4.3 {Error handling - too many arguments} -body {
    torch::set_grad_enabled 1 2
} -returnCodes error -result {Error in set_grad_enabled: Usage: torch::set_grad_enabled enabled}

;# Verify state is preserved across operations
test set_grad_enabled-5.1 {State preservation} {
    torch::set_grad_enabled 1
    set result1 [torch::is_grad_enabled]
    torch::set_grad_enabled 0
    set result2 [torch::is_grad_enabled]
    torch::set_grad_enabled 1
    set result3 [torch::is_grad_enabled]
    list $result1 $result2 $result3
} {1 0 1}

cleanupTests 