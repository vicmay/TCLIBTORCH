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

# Create test tensors
set a [torch::tensor_create {1 2 3 4} float32]
set b [torch::tensor_create {0 2 4 4} float32]

# -----------------------------------------------------------------------------
# Test 1: Basic functionality
# -----------------------------------------------------------------------------

test ge-1.1 {Positional syntax} {
    set res [torch::ge $a $b]
    expr {[string match "tensor*" $res]}
} {1}

test ge-1.2 {Named syntax} {
    set res [torch::ge -input1 $a -input2 $b]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 2: Error handling
# -----------------------------------------------------------------------------
# Missing args
test ge-2.1 {Missing args} {
    catch {torch::ge} msg
    expr {[string match "*Usage*" $msg]}
} {1}
test ge-2.2 {Invalid tensor handle} {
    catch {torch::ge bad $b} msg
    expr {[string match "*Invalid*" $msg]}
} {1}
test ge-2.3 {Unknown named param} {
    catch {torch::ge -foo $a -input2 $b} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

cleanupTests 