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

# =============================================================
# torch::initial_seed TEST SUITE
# =============================================================

# Test 1: Basic functionality (snake_case)
test initial_seed-1.1 {Basic call returns expected seed} {
    set seed [torch::initial_seed]
    expr {$seed == 2147483647}
} 1

# Test 2: CamelCase alias
test initial_seed-2.1 {CamelCase alias returns same value} {
    set seed1 [torch::initial_seed]
    set seed2 [torch::initialSeed]
    expr {$seed1 == $seed2}
} 1

# Test 3: Error handling - passing extraneous parameter
test initial_seed-3.1 {Error on extra parameter} {
    set rc [catch {torch::initial_seed extra_param} msg]
    expr {$rc == 1}
} 1

cleanupTests 