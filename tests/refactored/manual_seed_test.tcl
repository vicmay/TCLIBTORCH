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

# Test 1: Basic positional syntax (backward compatibility)
test manual_seed-1.1 {Basic positional syntax} {
    set result [torch::manual_seed 42]
    expr {$result eq "ok"}
} {1}

# Test 2: Named parameter syntax with -seed
test manual_seed-2.1 {Named parameter syntax with -seed} {
    set result [torch::manual_seed -seed 123]
    expr {$result eq "ok"}
} {1}

# Test 3: Alternative parameter name -s
test manual_seed-2.2 {Named parameter syntax with -s} {
    set result [torch::manual_seed -s 456]
    expr {$result eq "ok"}
} {1}

# Test 4: camelCase alias functionality
test manual_seed-3.1 {camelCase alias torch::manualSeed} {
    set result [torch::manualSeed 789]
    expr {$result eq "ok"}
} {1}

# Test 5: camelCase alias with named parameters
test manual_seed-3.2 {camelCase alias with named parameters} {
    set result [torch::manualSeed -seed 101112]
    expr {$result eq "ok"}
} {1}

# Test 6: Reproducibility test - same seed should produce same random tensors
test manual_seed-4.1 {Reproducibility with same seed} {
    # Set seed and create random tensor
    torch::manual_seed 42
    set tensor1 [torch::randn {2 3}]
    
    # Set same seed and create another random tensor
    torch::manual_seed 42
    set tensor2 [torch::randn {2 3}]
    
    # Check if tensors are approximately equal
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    
    # Convert to scalar and check if very small (numerical precision)
    set max_val [torch::tensor_item $max_diff]
    expr {$max_val < 1e-6}
} {1}

# Test 7: Different seeds produce different results
test manual_seed-4.2 {Different seeds produce different results} {
    # Set seed and create random tensor
    torch::manual_seed 42
    set tensor1 [torch::randn {2 3}]
    
    # Set different seed and create another random tensor
    torch::manual_seed 84
    set tensor2 [torch::randn {2 3}]
    
    # Check if tensors are different
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    
    # Convert to scalar and check if significantly different
    set max_val [torch::tensor_item $max_diff]
    expr {$max_val > 0.1}
} {1}

# Test 8: Large seed values
test manual_seed-4.3 {Large seed values} {
    set result [torch::manual_seed 2147483647]
    expr {$result eq "ok"}
} {1}

# Test 9: Zero seed value
test manual_seed-4.4 {Zero seed value} {
    set result [torch::manual_seed 0]
    expr {$result eq "ok"}
} {1}

# Test 10: Error handling - missing seed parameter
test manual_seed-5.1 {Error handling - missing seed parameter} {
    catch {torch::manual_seed} msg
    expr {[string match "*Required parameters missing*" $msg]}
} {1}

# Test 11: Error handling - invalid seed parameter
test manual_seed-5.2 {Error handling - invalid seed parameter} {
    catch {torch::manual_seed abc} msg
    expr {[string match "*Error*" $msg]}
} {1}

# Test 12: Error handling - negative seed
test manual_seed-5.3 {Error handling - negative seed} {
    catch {torch::manual_seed -1} msg
    expr {[string match "*manual_seed*" $msg] && [string match "*must*" $msg]}
} {1}

# Test 13: Error handling - unknown parameter
test manual_seed-5.4 {Error handling - unknown parameter} {
    catch {torch::manual_seed -unknown 42} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

# Test 14: Error handling - missing value for named parameter
test manual_seed-5.5 {Error handling - missing value for named parameter} {
    catch {torch::manual_seed -seed} msg
    expr {[string match "*pairs*" $msg]}
} {1}

# Test 15: Parameter order independence
test manual_seed-6.1 {Parameter order independence} {
    set result [torch::manual_seed -seed 555]
    expr {$result eq "ok"}
} {1}

cleanupTests 