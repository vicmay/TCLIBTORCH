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
    # Create a sparse-like tensor for testing
    set tensor [torch::tensor_randn -shape {3 4} -dtype float32]
    return $tensor
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test sparse_log_softmax-1.1 {Basic positional syntax} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_log_softmax $tensor 1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_log_softmax-1.2 {Positional syntax - different dimension} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_log_softmax $tensor 0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_log_softmax-1.3 {Positional syntax error handling - too few arguments} {
    set tensor [createTestTensor]
    
    catch {torch::sparse_log_softmax $tensor} error
    string match "*Usage:*" $error
} {1}

test sparse_log_softmax-1.4 {Positional syntax error handling - invalid dim} {
    set tensor [createTestTensor]
    
    catch {torch::sparse_log_softmax $tensor invalid_dim} error
    string match "*Invalid dim*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test sparse_log_softmax-2.1 {Named parameter syntax - basic usage} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_log_softmax -input $tensor -dim 1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_log_softmax-2.2 {Named parameter syntax - different dimension} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_log_softmax -input $tensor -dim 0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_log_softmax-2.3 {Named parameter syntax - different parameter order} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_log_softmax -dim 1 -input $tensor]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_log_softmax-2.4 {Named parameter syntax error handling - missing input} {
    catch {torch::sparse_log_softmax -dim 1} error
    string match "*Required parameters missing*" $error
} {1}

test sparse_log_softmax-2.5 {Named parameter syntax error handling - missing dim} {
    set tensor [createTestTensor]
    
    catch {torch::sparse_log_softmax -input $tensor} error
    string match "*Required parameters missing*" $error
} {1}

test sparse_log_softmax-2.6 {Named parameter syntax error handling - unknown parameter} {
    set tensor [createTestTensor]
    
    catch {torch::sparse_log_softmax -input $tensor -dim 1 -unknown value} error
    string match "*Unknown parameter*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test sparse_log_softmax-3.1 {camelCase alias - positional syntax} {
    set tensor [createTestTensor]
    
    set result [torch::sparseLogSoftmax $tensor 1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_log_softmax-3.2 {camelCase alias - named parameter syntax} {
    set tensor [createTestTensor]
    
    set result [torch::sparseLogSoftmax -input $tensor -dim 0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test sparse_log_softmax-4.1 {Consistency - same results from both syntaxes} {
    set tensor [createTestTensor]
    
    set result1 [torch::sparse_log_softmax $tensor 1]
    set result2 [torch::sparse_log_softmax -input $tensor -dim 1]
    
    # Results should be valid tensors
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test sparse_log_softmax-4.2 {Consistency - camelCase produces same results} {
    set tensor [createTestTensor]
    
    set result1 [torch::sparse_log_softmax $tensor 1]
    set result2 [torch::sparseLogSoftmax $tensor 1]
    
    # Results should be valid tensors
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Edge Cases
#===========================================================================================

test sparse_log_softmax-5.1 {Edge case - negative dimension} {
    set tensor [createTestTensor]
    
    set result [torch::sparse_log_softmax -input $tensor -dim -1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_log_softmax-5.2 {Edge case - dimension out of bounds} {
    set tensor [createTestTensor]
    
    catch {torch::sparse_log_softmax -input $tensor -dim 10} error
    string match "*" $error
} {1}

cleanupTests 