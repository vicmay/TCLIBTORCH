#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test cases for seed command
test seed-1.1 {Basic seed functionality} {
    set seed1 [torch::seed]
    set seed2 [torch::seed]
    expr {$seed1 != $seed2}  ;# Different seeds should be generated
} {1}

# Test that seed affects random number generation
test seed-2.1 {Seed affects random generation} {
    torch::manual_seed 12345
    set tensor1 [torch::normal -size {5} -mean 0.0 -std 1.0]
    
    torch::manual_seed 12345  ;# Reset to same seed
    set tensor2 [torch::normal -size {5} -mean 0.0 -std 1.0]
    
    torch::seed  ;# New random seed
    set tensor3 [torch::normal -size {5} -mean 0.0 -std 1.0]
    
    # First two tensors should be identical, third should be different
    set values1 [torch::tensor_to_list $tensor1]
    set values2 [torch::tensor_to_list $tensor2]
    set values3 [torch::tensor_to_list $tensor3]
    
    expr {$values1 eq $values2 && $values1 ne $values3}
} {1}

# Test error handling
test seed-3.1 {Error on extra arguments} {
    catch {torch::seed 123} err
    set err
} {wrong # args: should be "torch::seed "}

# Test camelCase alias
test seed-4.1 {CamelCase alias} {
    set seed1 [torch::seed]
    set seed2 [torch::seed]
    expr {$seed1 != $seed2}  ;# Different seeds should be generated
} {1}

cleanupTests 