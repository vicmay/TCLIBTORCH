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

# Essential tests (command already has dual syntax, adding camelCase)
test broadcast_shapes-1.1 {Basic functionality} {
    set result [torch::broadcast_shapes {2 3} {1 3}]
    expr {$result ne ""}
} {1}

test broadcast_shapes-3.1 {CamelCase alias} {
    set result [torch::broadcastShapes {5} {3 5}]
    expr {$result ne ""}
} {1}

cleanupTests 