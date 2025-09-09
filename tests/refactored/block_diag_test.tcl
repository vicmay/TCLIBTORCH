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

# Essential tests for both syntaxes (command already has dual syntax)
test block_diag-1.1 {Basic functionality} {
    set t1 [torch::tensor_create {{1 2} {3 4}} float32]
    set t2 [torch::tensor_create {{5 6}} float32]
    set result [torch::block_diag $t1 $t2]
    expr {$result ne ""}
} {1}

test block_diag-3.1 {CamelCase alias} {
    set t1 [torch::tensor_create {{1 0} {0 1}} float32]
    set t2 [torch::tensor_create {{2}} float32]
    set result [torch::blockDiag $t1 $t2]
    expr {$result ne ""}
} {1}

test block_diag-4.1 {Multiple tensors} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3} float32]
    set t3 [torch::tensor_create {4 5} float32]
    set result [torch::block_diag $t1 $t2 $t3]
    expr {$result ne ""}
} {1}

cleanupTests 