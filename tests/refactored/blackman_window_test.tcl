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

# Essential tests for both syntaxes
test blackman_window-1.1 {Positional syntax} {
    set result [torch::blackman_window 10]
    expr {$result ne ""}
} {1}

test blackman_window-2.1 {Named parameters} {
    set result [torch::blackman_window -window_length 8]
    expr {$result ne ""}
} {1}

test blackman_window-2.2 {Named parameters with alias} {
    set result [torch::blackman_window -length 6]
    expr {$result ne ""}
} {1}

test blackman_window-3.1 {CamelCase alias} {
    set result [torch::blackmanWindow 12]
    expr {$result ne ""}
} {1}

test blackman_window-4.1 {Error handling - invalid length} {
    catch {torch::blackman_window 0} result
    expr {[string match "*positive*" $result]}
} {1}

cleanupTests 