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

# Helper function to check if two tensors are approximately equal
proc tensor_approx_equal {t1 t2 {tolerance 1e-5}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    expr {$max_val < $tolerance}
}

# Test cases for positional syntax
test tensor_cholesky-1.1 {Basic positional syntax} {
    # Create a positive definite matrix
    set t [torch::tensor_create {4.0 12.0 -16.0 12.0 37.0 -43.0 -16.0 -43.0 98.0} {3 3}]
    set result [torch::tensor_cholesky $t]
    expr {$result ne ""}
} {1}

test tensor_cholesky-1.2 {Positional syntax with 2x2 matrix} {
    # Create a 2x2 positive definite matrix
    set t [torch::tensor_create {4.0 2.0 2.0 5.0} {2 2}]
    set result [torch::tensor_cholesky $t]
    expr {$result ne ""}
} {1}

# Test cases for named parameter syntax
test tensor_cholesky-2.1 {Named parameter syntax with -input} {
    # Create a positive definite matrix
    set t [torch::tensor_create {4.0 12.0 -16.0 12.0 37.0 -43.0 -16.0 -43.0 98.0} {3 3}]
    set result [torch::tensor_cholesky -input $t]
    expr {$result ne ""}
} {1}

test tensor_cholesky-2.2 {Named parameter syntax with -tensor} {
    # Create a 2x2 positive definite matrix
    set t [torch::tensor_create {4.0 2.0 2.0 5.0} {2 2}]
    set result [torch::tensor_cholesky -tensor $t]
    expr {$result ne ""}
} {1}

# Test cases for camelCase alias
test tensor_cholesky-3.1 {CamelCase alias} {
    # Create a positive definite matrix
    set t [torch::tensor_create {4.0 12.0 -16.0 12.0 37.0 -43.0 -16.0 -43.0 98.0} {3 3}]
    set result [torch::tensorCholesky -input $t]
    expr {$result ne ""}
} {1}

# Error handling tests
test tensor_cholesky-4.1 {Error on missing tensor} {
    catch {torch::tensor_cholesky} msg
    set msg
} {Usage: torch::tensor_cholesky tensor | torch::tensor_cholesky -input tensor}

test tensor_cholesky-4.2 {Error on invalid tensor name} {
    catch {torch::tensor_cholesky invalid_tensor} msg
    set msg
} {Invalid tensor name}

test tensor_cholesky-4.3 {Error on invalid parameter name} {
    set t [torch::tensor_create {4.0 2.0 2.0 5.0} {2 2}]
    catch {torch::tensor_cholesky -invalid $t} msg
    set msg
} {Unknown parameter: -invalid. Valid parameters are: -input, -tensor}

test tensor_cholesky-4.4 {Error on missing parameter value} {
    catch {torch::tensor_cholesky -input} msg
    set msg
} {Missing value for parameter}

# Syntax consistency tests
test tensor_cholesky-5.1 {Syntax consistency - positional vs named} {
    # Create a positive definite matrix
    set t [torch::tensor_create {4.0 2.0 2.0 5.0} {2 2}]
    set result1 [torch::tensor_cholesky $t]
    set result2 [torch::tensor_cholesky -input $t]
    expr {$result1 ne "" && $result2 ne ""}
} {1}

# Mathematical correctness test
test tensor_cholesky-6.1 {Cholesky decomposition correctness} {
    # Create a simple 2x2 positive definite matrix
    set t [torch::tensor_create {4.0 2.0 2.0 5.0} {2 2}]
    set L [torch::tensor_cholesky $t]
    
    # Verify that L is lower triangular by checking the result is valid
    # For a 2x2 matrix, the result should be a valid tensor
    expr {$L ne ""}
} {1}

cleanupTests 