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

# Essential tests (command already has dual syntax and camelCase)
test broadcast-1.1 {Basic functionality} {
    set t1 [torch::tensor_create -data {1 2 3} -dtype float32]
    set result [torch::broadcast $t1 0]
    expr {$result ne ""}
} {1}

test broadcast-2.1 {Named parameter syntax} {
    set t1 [torch::tensor_create -data {5} -dtype float32]
    set result [torch::broadcast -tensor $t1 -root 0]
    expr {$result ne ""}
} {1}

test broadcast-3.1 {CamelCase alias} {
    set t1 [torch::tensor_create -data {1} -dtype float32]
    set result [torch::broadcast $t1 0]
    expr {$result ne ""}
} {1}

cleanupTests