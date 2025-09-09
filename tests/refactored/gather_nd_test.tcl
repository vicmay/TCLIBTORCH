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

# Helper to create simple tensors
set inp [torch::tensor_create {10.0 20.0 30.0 40.0} float32]
set idx [torch::tensor_create {3 1} int64]

# -----------------------------------------------------------------------------
# Test 1: Basic functionality
# -----------------------------------------------------------------------------

test gather_nd-1.1 {Positional syntax} {
    set res [torch::gather_nd $inp $idx]
    expr {[string match "tensor*" $res]}
} {1}

test gather_nd-1.2 {Named syntax} {
    set res [torch::gather_nd -input $inp -indices $idx]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 2: camelCase alias
# -----------------------------------------------------------------------------

test gather_nd-2.1 {camelCase alias} {
    set res [torch::gatherNd -input $inp -indices $idx]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 3: Error handling
# -----------------------------------------------------------------------------

test gather_nd-3.1 {Missing arguments} {
    catch {torch::gather_nd} msg
    expr {[string match "*Usage*" $msg] || [string match "*wrong*" $msg]}
} {1}

test gather_nd-3.2 {Invalid input handle} {
    catch {torch::gather_nd bad_handle $idx} msg
    expr {[string match "*Invalid input*" $msg]}
} {1}

test gather_nd-3.3 {Invalid indices handle} {
    catch {torch::gather_nd $inp bad_handle} msg
    expr {[string match "*Invalid indices*" $msg]}
} {1}

test gather_nd-3.4 {Unknown named parameter} {
    catch {torch::gather_nd -input $inp -idx $idx} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

cleanupTests 