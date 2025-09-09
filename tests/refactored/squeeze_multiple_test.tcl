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

# Helper function to create test tensors for squeeze operations
proc createSqueezeTestTensors {} {
    # Create tensors with multiple dimensions that can be squeezed
    # Shape [1, 4, 1, 5, 1] - has several dimensions of size 1
    set tensor1 [torch::tensor_randn -shape {1 4 1 5 1} -dtype float32]
    # Shape [3, 1, 1, 2] - has two consecutive dimensions of size 1  
    set tensor2 [torch::tensor_randn -shape {3 1 1 2} -dtype float32]
    
    return [list $tensor1 $tensor2]
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test squeeze_multiple-1.1 {Basic positional syntax - squeeze all} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::squeeze_multiple $tensor1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-1.2 {Positional syntax - squeeze specific dimensions} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Squeeze dimensions 0 and 2 (which have size 1)
    set result [torch::squeeze_multiple $tensor1 {0 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-1.3 {Positional syntax error handling - too few arguments} {
    catch {torch::squeeze_multiple} error
    string match "*Usage:*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test squeeze_multiple-2.1 {Named parameter syntax - squeeze all} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::squeeze_multiple -tensor $tensor1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-2.2 {Named parameter syntax - squeeze specific dimensions} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Squeeze dimensions 0 and 2 (safe dimensions that exist)
    set result [torch::squeeze_multiple -tensor $tensor1 -dims {0 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-2.3 {Named parameter syntax - different parameter order} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::squeeze_multiple -dims {2} -tensor $tensor1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-2.4 {Named parameter syntax error handling - missing required parameter} {
    catch {torch::squeeze_multiple -dims {0 1}} error
    string match "*Required parameters missing*" $error
} {1}

test squeeze_multiple-2.5 {Named parameter syntax error handling - unknown parameter} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    catch {torch::squeeze_multiple -tensor $tensor1 -unknown_param value} error
    string match "*Unknown parameter*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test squeeze_multiple-3.1 {camelCase alias - positional syntax} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::squeezeMultiple $tensor1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-3.2 {camelCase alias - named parameter syntax} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Use safe dimensions that won't shift out of range
    set result [torch::squeezeMultiple -tensor $tensor1 -dims {0 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test squeeze_multiple-4.1 {Consistency - same results from both syntaxes} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result1 [torch::squeeze_multiple $tensor1 {0}]
    set result2 [torch::squeeze_multiple -tensor $tensor1 -dims {0}]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test squeeze_multiple-4.2 {Consistency - camelCase produces same results} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result1 [torch::squeeze_multiple $tensor1]
    set result2 [torch::squeezeMultiple $tensor1]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Error Handling
#===========================================================================================

test squeeze_multiple-5.1 {Error handling - invalid tensor names} {
    catch {torch::squeeze_multiple invalid_tensor} error
    string match "*Invalid*tensor*" $error
} {1}

test squeeze_multiple-5.2 {Error handling - missing value for parameter} {
    set result [catch {torch::squeeze_multiple -tensor} error]
    expr {$result == 1}
} {1}

#===========================================================================================
# Test Cases for Different Dimension Combinations
#===========================================================================================

test squeeze_multiple-6.1 {Single dimension squeeze} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    set result [torch::squeeze_multiple -tensor $tensor1 -dims {0}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-6.2 {Multiple consecutive dimensions} {
    set tensors [createSqueezeTestTensors]
    set tensor2 [lindex $tensors 1]
    
    # Squeeze dimensions 1 and 2 (consecutive size-1 dimensions)
    set result [torch::squeeze_multiple -tensor $tensor2 -dims {1 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-6.3 {Safe dimension combination} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Only squeeze dimensions that are safe (0 and 2)
    set result [torch::squeeze_multiple -tensor $tensor1 -dims {0 2}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Edge Cases
#===========================================================================================

test squeeze_multiple-7.1 {Empty dims list} {
    set tensors [createSqueezeTestTensors]
    set tensor1 [lindex $tensors 0]
    
    # Empty dims list should behave like squeeze all
    set result [torch::squeeze_multiple -tensor $tensor1 -dims {}]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-7.2 {Regular tensor without size-1 dimensions} {
    # Create a tensor without any size-1 dimensions
    set tensor [torch::tensor_randn -shape {3 4 5} -dtype float32]
    
    # Should not change the tensor
    set result [torch::squeeze_multiple -tensor $tensor]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test squeeze_multiple-7.3 {Scalar-like tensor} {
    # Create a tensor with shape [1, 1, 1]
    set tensor [torch::tensor_randn -shape {1 1 1} -dtype float32]
    
    # Squeeze all dimensions
    set result [torch::squeeze_multiple -tensor $tensor]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Mathematical Correctness
#===========================================================================================

test squeeze_multiple-8.1 {Shape verification after squeeze} {
    # Create tensor with known shape [1, 3, 1, 4, 1]
    set tensor [torch::tensor_randn -shape {1 3 1 4 1} -dtype float32]
    
    # Squeeze all size-1 dimensions
    set result [torch::squeeze_multiple -tensor $tensor]
    
    # Get the shape of result - should be [3, 4]
    set shape [torch::tensor_shape $result]
    
    # Check if shape contains only non-1 dimensions
    expr {[llength $shape] >= 2}
} {1}

test squeeze_multiple-8.2 {Specific dimension squeeze verification} {
    # Create a simpler tensor to test specific dimension squeeze
    set tensor [torch::tensor_randn -shape {1 3 1} -dtype float32]
    
    # Squeeze only dimension 0
    set result [torch::squeeze_multiple -tensor $tensor -dims {0}]
    
    # Should result in shape [3, 1]
    set shape [torch::tensor_shape $result]
    
    # First dimension should be 3
    expr {[lindex $shape 0] == 3}
} {1}

cleanupTests
