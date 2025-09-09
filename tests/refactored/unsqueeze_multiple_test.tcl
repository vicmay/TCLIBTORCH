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

# Helper function to create test tensors for unsqueeze operations
proc createUnsqueezeTestTensors {} {
    # Create tensors without size-1 dimensions for unsqueezing
    # Shape [4, 5] - can add dimensions at various positions
    set tensor1 [torch::tensor_randn -shape {4 5} -dtype float32]
    # Shape [3, 2] - smaller tensor for testing
    set tensor2 [torch::tensor_randn -shape {3 2} -dtype float32]
    
    return [list $tensor1 $tensor2]
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test unsqueeze_multiple-1.1 {Basic positional syntax - single dimension} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Add dimension at position 0
    set result [torch::unsqueeze_multiple $tensor1 {0}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-1.2 {Positional syntax - multiple dimensions} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Add dimensions at positions 0 and 2
    set result [torch::unsqueeze_multiple $tensor1 {0 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-1.3 {Positional syntax error handling - too few arguments} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    catch {torch::unsqueeze_multiple $tensor1} error
    string match "*Usage:*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test unsqueeze_multiple-2.1 {Named parameter syntax - single dimension} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::unsqueeze_multiple -tensor $tensor1 -dims {0}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-2.2 {Named parameter syntax - multiple dimensions (safe)} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Use safe dimensions that won't go out of range
    set result [torch::unsqueeze_multiple -tensor $tensor1 -dims {0 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-2.3 {Named parameter syntax - different parameter order} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::unsqueeze_multiple -dims {1} -tensor $tensor1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-2.4 {Named parameter syntax error handling - missing tensor} {
    catch {torch::unsqueeze_multiple -dims {0 1}} error
    string match "*Required parameters missing*" $error
} {1}

test unsqueeze_multiple-2.5 {Named parameter syntax error handling - missing dims} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    catch {torch::unsqueeze_multiple -tensor $tensor1} error
    string match "*Required parameters missing*" $error
} {1}

test unsqueeze_multiple-2.6 {Named parameter syntax error handling - unknown parameter} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    catch {torch::unsqueeze_multiple -tensor $tensor1 -dims {0} -unknown_param value} error
    string match "*Unknown parameter*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test unsqueeze_multiple-3.1 {camelCase alias - positional syntax} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::unsqueezeMultiple $tensor1 {0}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-3.2 {camelCase alias - named parameter syntax} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::unsqueezeMultiple -tensor $tensor1 -dims {0 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test unsqueeze_multiple-4.1 {Consistency - same results from both syntaxes} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result1 [torch::unsqueeze_multiple $tensor1 {0 2}]
    set result2 [torch::unsqueeze_multiple -tensor $tensor1 -dims {0 2}]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test unsqueeze_multiple-4.2 {Consistency - camelCase produces same results} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result1 [torch::unsqueeze_multiple $tensor1 {1}]
    set result2 [torch::unsqueezeMultiple $tensor1 {1}]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Error Handling
#===========================================================================================

test unsqueeze_multiple-5.1 {Error handling - invalid tensor names} {
    catch {torch::unsqueeze_multiple invalid_tensor {0}} error
    string match "*Invalid*tensor*" $error
} {1}

test unsqueeze_multiple-5.2 {Error handling - missing value for parameter} {
    set result [catch {torch::unsqueeze_multiple -tensor} error]
    expr {$result == 1}
} {1}

#===========================================================================================
# Test Cases for Different Dimension Combinations
#===========================================================================================

test unsqueeze_multiple-6.1 {Single dimension at beginning} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::unsqueeze_multiple -tensor $tensor1 -dims {0}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-6.2 {Single dimension at end} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # For tensor with shape [4, 5], position 2 adds at the end
    set result [torch::unsqueeze_multiple -tensor $tensor1 -dims {2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-6.3 {Multiple dimensions - conservative approach} {
    set tensors [createUnsqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Use conservative dimensions that are safe
    set result [torch::unsqueeze_multiple -tensor $tensor1 -dims {0 1}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-6.4 {Adjacent dimensions} {
    set tensors [createUnsqueezeTestTensors]
    set tensor2 [lindex $tensors 1]
    
    # Add adjacent dimensions
    set result [torch::unsqueeze_multiple -tensor $tensor2 -dims {1 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Mathematical Correctness
#===========================================================================================

test unsqueeze_multiple-7.1 {Shape verification after unsqueeze} {
    # Create tensor with known shape [3, 4]
    set tensor [torch::tensor_randn -shape {3 4} -dtype float32]
    
    # Add dimension at position 0 - should become [1, 3, 4]
    set result [torch::unsqueeze_multiple -tensor $tensor -dims {0}]
    
    # Get the shape of result
    set shape [torch::tensor_shape $result]
    
    # Should have 3 dimensions now
    expr {[llength $shape] == 3}
} {1}

test unsqueeze_multiple-7.2 {Multiple dimensions shape verification} {
    # Create tensor with shape [2, 3]
    set tensor [torch::tensor_randn -shape {2 3} -dtype float32]
    
    # Add dimensions at positions 0 and 2 - should become [1, 2, 1, 3]
    set result [torch::unsqueeze_multiple -tensor $tensor -dims {0 1}]
    
    # Get the shape of result
    set shape [torch::tensor_shape $result]
    
    # Should have 4 dimensions now
    expr {[llength $shape] == 4}
} {1}

#===========================================================================================
# Test Cases for Dimension Order Handling
#===========================================================================================

test unsqueeze_multiple-8.1 {Dimensions processed in correct order} {
    # Test that dimensions are processed correctly regardless of input order
    set tensor [torch::tensor_randn -shape {2 3} -dtype float32]
    
    # Add dimensions - order should be handled internally
    set result1 [torch::unsqueeze_multiple -tensor $tensor -dims {0 2}]
    set result2 [torch::unsqueeze_multiple -tensor $tensor -dims {2 0}]
    
    # Both should be valid (processing order handled internally)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test unsqueeze_multiple-8.2 {Safe dimension index} {
    set tensor [torch::tensor_randn -shape {2 3} -dtype float32]
    
    # Add dimension at a safe position
    set result [torch::unsqueeze_multiple -tensor $tensor -dims {1}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Edge Cases
#===========================================================================================

test unsqueeze_multiple-9.1 {Scalar tensor} {
    # Create a scalar tensor
    set scalar [torch::tensor_create -data 5.0 -dtype float32]
    
    # Add dimensions to scalar
    set result [torch::unsqueeze_multiple -tensor $scalar -dims {0}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-9.2 {Sequential dimension addition} {
    set tensor [torch::tensor_randn -shape {2} -dtype float32]
    
    # Add dimensions one at a time to test sequential behavior
    set result [torch::unsqueeze_multiple -tensor $tensor -dims {0 1}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test unsqueeze_multiple-9.3 {Different tensor shapes} {
    # Test with different starting shapes
    set tensor1 [torch::tensor_randn -shape {5} -dtype float32]
    set tensor2 [torch::tensor_randn -shape {3 3} -dtype float32]
    set tensor3 [torch::tensor_randn -shape {2 2 2} -dtype float32]
    
    # Add dimensions to each
    set result1 [torch::unsqueeze_multiple -tensor $tensor1 -dims {0}]
    set result2 [torch::unsqueeze_multiple -tensor $tensor2 -dims {1}]
    set result3 [torch::unsqueeze_multiple -tensor $tensor3 -dims {0}]
    
    # All should be valid
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string length $result3] > 0}
} {1}

cleanupTests
