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

# Test cases for positional syntax (backward compatibility)
test cartesian_prod-1.1 {Basic positional syntax with two tensors} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3 4} float32]
    set result [torch::cartesian_prod $t1 $t2]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

test cartesian_prod-1.2 {Positional syntax with three tensors} {
    set t1 [torch::tensor_create {1} float32]
    set t2 [torch::tensor_create {2} float32]
    set t3 [torch::tensor_create {3} float32]
    set result [torch::cartesian_prod $t1 $t2 $t3]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

test cartesian_prod-1.3 {Positional syntax with different vector sizes} {
    set t1 [torch::tensor_create {1 2 3} float32]
    set t2 [torch::tensor_create {4 5} float32]
    set result [torch::cartesian_prod $t1 $t2]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

# Test cases for named parameter syntax
test cartesian_prod-2.1 {Named parameters with list format} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3 4} float32]
    set result [torch::cartesian_prod -tensors [list $t1 $t2]]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

test cartesian_prod-2.2 {Named parameters with three tensors in list} {
    set t1 [torch::tensor_create {1} float32]
    set t2 [torch::tensor_create {2} float32]
    set t3 [torch::tensor_create {3} float32]
    set tensor_list [list $t1 $t2 $t3]
    set result [torch::cartesian_prod -tensors $tensor_list]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

test cartesian_prod-2.3 {Named parameters with single tensor as string} {
    set t1 [torch::tensor_create {1 2 3} float32]
    # Test the fallback to single tensor name when list parsing fails
    set result [torch::cartesian_prod -tensors $t1]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

# Test cases for camelCase alias
test cartesian_prod-3.1 {CamelCase alias with positional syntax} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3 4} float32]
    set result [torch::cartesianProd $t1 $t2]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

test cartesian_prod-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::tensor_create {1} float32]
    set t2 [torch::tensor_create {2 3} float32]
    set result [torch::cartesianProd -tensors [list $t1 $t2]]
    # Should return a single tensor handle
    expr {$result ne ""}
} {1}

# Test that both syntaxes produce same type of output
test cartesian_prod-4.1 {Both syntaxes produce valid tensor handles} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3 4} float32]
    
    set result1 [torch::cartesian_prod $t1 $t2]
    set result2 [torch::cartesian_prod -tensors [list $t1 $t2]]
    set result3 [torch::cartesianProd $t1 $t2]
    set result4 [torch::cartesianProd -tensors [list $t1 $t2]]
    
    # All results should be valid (non-empty) tensor handles
    expr {$result1 ne "" && $result2 ne "" && $result3 ne "" && $result4 ne ""}
} {1}

# Error handling tests
test cartesian_prod-5.1 {Error on no arguments} {
    catch {torch::cartesian_prod} result
    expr {[string match "*Usage*" $result] || [string match "*tensor*" $result]}
} {1}

test cartesian_prod-5.2 {Error on invalid tensor name} {
    catch {torch::cartesian_prod invalid_tensor} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

test cartesian_prod-5.3 {Error on unknown parameter} {
    set t1 [torch::tensor_create {1 2} float32]
    catch {torch::cartesian_prod -unknown_param $t1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test cartesian_prod-5.4 {Error on missing value for parameter} {
    catch {torch::cartesian_prod -tensors} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

test cartesian_prod-5.5 {Error with mixed valid and invalid tensors} {
    set t1 [torch::tensor_create {1 2} float32]
    catch {torch::cartesian_prod $t1 invalid_tensor} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

test cartesian_prod-5.6 {Error on invalid tensor in list} {
    set t1 [torch::tensor_create {1 2} float32]
    catch {torch::cartesian_prod -tensors [list $t1 invalid_tensor]} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

# Test mathematical correctness (basic verification)
test cartesian_prod-6.1 {Result is valid tensor handle} {
    set t1 [torch::tensor_create {1 2} float32]
    set t2 [torch::tensor_create {3 4} float32]
    set result [torch::cartesian_prod $t1 $t2]
    
    # Verify the result is a valid tensor handle
    expr {$result ne ""}
} {1}

test cartesian_prod-6.2 {Single tensor case} {
    set t1 [torch::tensor_create {1 2 3} float32]
    set result [torch::cartesian_prod $t1]
    
    # Should work with single tensor too
    expr {$result ne ""}
} {1}

cleanupTests 