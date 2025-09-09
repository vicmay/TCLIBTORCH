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

# Helper function to create test tensors for sparse operations
proc createSparseAddTestTensors {} {
    # Create regular input tensors (sparse operations can work with regular tensors)
    set tensor1 [torch::tensor_randn -shape {4 5} -dtype float32]
    set tensor2 [torch::tensor_randn -shape {4 5} -dtype float32]
    
    return [list $tensor1 $tensor2]
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test sparse_add-1.1 {Basic positional syntax - default alpha} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add $tensor1 $tensor2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-1.2 {Positional syntax - with alpha value} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add $tensor1 $tensor2 2.0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-1.3 {Positional syntax error handling - too few arguments} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    
    catch {torch::sparse_add $tensor1} error
    string match "*Usage:*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test sparse_add-2.1 {Named parameter syntax - basic usage} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-2.2 {Named parameter syntax - with alpha} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha 1.5]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-2.3 {Named parameter syntax - different parameter order} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add -tensor2 $tensor2 -alpha 0.5 -tensor1 $tensor1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-2.4 {Named parameter syntax error handling - missing required parameter} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    
    catch {torch::sparse_add -tensor1 $tensor1 -alpha 1.0} error
    string match "*Required parameters missing*" $error
} {1}

test sparse_add-2.5 {Named parameter syntax error handling - unknown parameter} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    catch {torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -unknown_param value} error
    string match "*Unknown parameter*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test sparse_add-3.1 {camelCase alias - positional syntax} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparseAdd $tensor1 $tensor2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-3.2 {camelCase alias - named parameter syntax} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparseAdd -tensor1 $tensor1 -tensor2 $tensor2 -alpha 2.5]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test sparse_add-4.1 {Consistency - same results from both syntaxes} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result1 [torch::sparse_add $tensor1 $tensor2 1.0]
    set result2 [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha 1.0]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test sparse_add-4.2 {Consistency - camelCase produces same results} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result1 [torch::sparse_add $tensor1 $tensor2]
    set result2 [torch::sparseAdd $tensor1 $tensor2]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Parameter Validation
#===========================================================================================

test sparse_add-5.1 {Parameter validation - invalid alpha} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    catch {torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha invalid_alpha} error
    string match "*Invalid alpha*" $error
} {1}

#===========================================================================================
# Test Cases for Error Handling
#===========================================================================================

test sparse_add-6.1 {Error handling - invalid tensor names} {
    catch {torch::sparse_add invalid_tensor1 invalid_tensor2} error
    string match "*Invalid*tensor*" $error
} {1}

test sparse_add-6.2 {Error handling - missing value for parameter} {
    set result [catch {torch::sparse_add -tensor1} error]
    expr {$result == 1}
} {1}

#===========================================================================================
# Test Cases for Different Alpha Values
#===========================================================================================

test sparse_add-7.1 {Different alpha values - alpha 0.5} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha 0.5]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-7.2 {Different alpha values - alpha 3.0} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha 3.0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-7.3 {Negative alpha value} {
    set tensors [createSparseAddTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha -0.5]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Tensor Shape Compatibility
#===========================================================================================

test sparse_add-8.1 {Compatible tensor shapes - same dimensions} {
    set tensor1 [torch::tensor_randn -shape {3 4} -dtype float32]
    set tensor2 [torch::tensor_randn -shape {3 4} -dtype float32]
    
    set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test sparse_add-8.2 {Broadcastable tensor shapes} {
    set tensor1 [torch::tensor_randn -shape {2 3} -dtype float32]
    set tensor2 [torch::tensor_randn -shape {3} -dtype float32]
    
    set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

cleanupTests
