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
test set_flush_denormal-1.1 {Basic positional syntax - enable} {
    torch::set_flush_denormal 1
} {denormal_flushing_enabled}

test set_flush_denormal-1.2 {Basic positional syntax - disable} {
    torch::set_flush_denormal 0
} {denormal_flushing_disabled}

;# Test cases for named syntax
test set_flush_denormal-2.1 {Named parameter syntax - enable} {
    torch::set_flush_denormal -enabled 1
} {denormal_flushing_enabled}

test set_flush_denormal-2.2 {Named parameter syntax - disable} {
    torch::set_flush_denormal -enabled 0
} {denormal_flushing_disabled}

;# Test cases for camelCase alias
test set_flush_denormal-3.1 {CamelCase alias - enable} {
    torch::setFlushDenormal -enabled 1
} {denormal_flushing_enabled}

test set_flush_denormal-3.2 {CamelCase alias - disable} {
    torch::setFlushDenormal -enabled 0
} {denormal_flushing_disabled}

;# Error handling tests
test set_flush_denormal-4.1 {Error handling - missing argument} -body {
    torch::set_flush_denormal
} -returnCodes error -result {Error in set_flush_denormal: wrong # args: should be "torch::set_flush_denormal enabled" or "torch::set_flush_denormal -enabled value"}

test set_flush_denormal-4.2 {Error handling - invalid argument type} -body {
    torch::set_flush_denormal invalid
} -returnCodes error -result {Error in set_flush_denormal: expected integer for enabled parameter}

cleanupTests 