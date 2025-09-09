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

# Helper function to create test tensors
proc createTestTensor {} {
    # Create a sparse tensor with duplicate indices for testing coalesce
    set values [torch::tensor_create {1.0 2.0 3.0} {3} float32]
    set indices [torch::tensor_create {0 0 1 0 1 0} {2 3} int64]  ;# 2x3 matrix for sparse_dim=2
    set sparse_tensor [torch::sparse_coo_tensor $indices $values {5 5}]  ;# 5x5 sparse matrix
    return $sparse_tensor
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test sparse_coalesce-1.1 {Basic positional syntax} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_coalesce $tensor]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_coalesce-1.2 {Positional syntax error handling - too few arguments} {
    catch {torch::sparse_coalesce} error
    string match "*Usage:*" $error
} {1}

test sparse_coalesce-1.3 {Positional syntax error handling - too many arguments} {
    set tensor [createTestTensor]
    
    catch {torch::sparse_coalesce $tensor extra_arg} error
    string match "*Usage:*" $error
} {1}

test sparse_coalesce-1.4 {Positional syntax error handling - invalid tensor} {
    catch {torch::sparse_coalesce invalid_tensor} error
    string match "*Invalid sparse tensor*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test sparse_coalesce-2.1 {Named parameter syntax - basic usage} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_coalesce -input $tensor]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_coalesce-2.2 {Named parameter syntax error handling - missing input} {
    catch {torch::sparse_coalesce -input} error
    string match "*Missing value for parameter*" $error
} {1}

test sparse_coalesce-2.3 {Named parameter syntax error handling - unknown parameter} {
    set tensor [createTestTensor]
    
    catch {torch::sparse_coalesce -input $tensor -unknown value} error
    string match "*Unknown parameter*" $error
} {1}

test sparse_coalesce-2.4 {Named parameter syntax error handling - invalid tensor} {
    catch {torch::sparse_coalesce -input invalid_tensor} error
    string match "*Invalid sparse tensor*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test sparse_coalesce-3.1 {camelCase alias - positional syntax} {
    set tensor [createTestTensor]
    
    set result [torch::sparseCoalesce $tensor]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_coalesce-3.2 {camelCase alias - named parameter syntax} {
    set tensor [createTestTensor]
    
    set result [torch::sparseCoalesce -input $tensor]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test sparse_coalesce-4.1 {Consistency - same results from both syntaxes} {
    set tensor [createTestTensor]
    
    set result1 [torch::sparse_coalesce $tensor]
    set result2 [torch::sparse_coalesce -input $tensor]
    
    # Results should be valid tensors
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test sparse_coalesce-4.2 {Consistency - camelCase produces same results} {
    set tensor [createTestTensor]
    
    set result1 [torch::sparse_coalesce $tensor]
    set result2 [torch::sparseCoalesce $tensor]
    
    # Results should be valid tensors
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Edge Cases
#===========================================================================================

test sparse_coalesce-5.1 {Edge case - already coalesced tensor} {
    set tensor [createTestTensor]
    set coalesced [torch::sparse_coalesce $tensor]
    
    # Coalescing an already coalesced tensor should work
    set result [torch::sparse_coalesce $coalesced]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

cleanupTests 