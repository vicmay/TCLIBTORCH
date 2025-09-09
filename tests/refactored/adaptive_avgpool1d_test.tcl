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
test adaptive_avgpool1d-1.1 {Basic positional syntax} {
    # Create a 1D tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    # Reshape to (batch, channels, length)
    set reshaped [torch::tensor_reshape $tensor {1 1 6}]
    set result [torch::adaptive_avgpool1d $reshaped 3]
    
    # Result should be a valid tensor handle
    expr {[string length $result] > 0}
} {1}

# Test 2: Named parameter syntax
test adaptive_avgpool1d-2.1 {Named parameter syntax} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {1 1 4}]
    set result [torch::adaptive_avgpool1d -input $reshaped -output_size 2]
    
    # Result should be a valid tensor handle
    expr {[string length $result] > 0}
} {1}

# Test 3: Alternative parameter names
test adaptive_avgpool1d-2.2 {Alternative parameter names} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {2 1 4}]
    set result [torch::adaptive_avgpool1d -tensor $reshaped -outputSize 2]
    
    expr {[string length $result] > 0}
} {1}

# Test 4: camelCase alias
test adaptive_avgpool1d-3.1 {camelCase alias syntax} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {1 1 4}]
    set result [torch::adaptiveAvgpool1d -input $reshaped -output_size 2]
    
    expr {[string length $result] > 0}
} {1}

# Test 5: Both syntaxes produce same result
test adaptive_avgpool1d-4.1 {Both syntaxes produce same result} {
    # Create identical tensors
    set tensor1 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set matrix1 [torch::tensor_reshape $tensor1 {1 1 6}]
    
    set tensor2 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set matrix2 [torch::tensor_reshape $tensor2 {1 1 6}]
    
    # Test positional syntax
    set result1 [torch::adaptive_avgpool1d $matrix1 3]
    
    # Test named syntax  
    set result2 [torch::adaptive_avgpool1d -input $matrix2 -output_size 3]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    string equal $shape1 $shape2
} {1}

# Test 6: Error handling - missing parameters (named syntax)
test adaptive_avgpool1d-5.1 {Error handling - missing input parameter} {
    set result [catch {torch::adaptive_avgpool1d -output_size 2} error]
    expr {$result == 1 && [string match "*Required parameters missing*" $error]}
} {1}

# Test 7: Error handling - wrong number of positional args
test adaptive_avgpool1d-5.2 {Error handling - wrong number of positional args} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu false]
    set result [catch {torch::adaptive_avgpool1d $tensor} error]
    expr {$result == 1}
} {1}

# Test 8: Error handling - invalid tensor name
test adaptive_avgpool1d-5.3 {Error handling - invalid tensor name} {
    set result [catch {torch::adaptive_avgpool1d -input "invalid_tensor" -output_size 2} error]
    expr {$result == 1 && [string match "*Invalid*tensor*" $error]}
} {1}

# Test 9: Error handling - unknown parameter
test adaptive_avgpool1d-5.4 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create {1.0 2.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {1 1 2}]
    set result [catch {torch::adaptive_avgpool1d -input $reshaped -badparam value -output_size 1} error]
    expr {$result == 1 && [string match "*Unknown parameter*" $error]}
} {1}

# Test 10: Output size reduction
test adaptive_avgpool1d-6.1 {Output size reduction} {
    # Create a longer sequence
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}
    set tensor [torch::tensor_create $data float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {1 1 8}]
    set result [torch::adaptive_avgpool1d -input $reshaped -output_size 4]
    
    # Should reduce from length 8 to length 4
    set shape [torch::tensor_shape $result]
    string equal $shape {1 1 4}
} {1}

# Test 11: Batch processing
test adaptive_avgpool1d-7.1 {Batch processing} {
    # Create a batch of sequences
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create $data float32 cpu false]
    # 2 batches, 2 channels, 3 length
    set reshaped [torch::tensor_reshape $tensor {2 2 3}]
    set result [torch::adaptive_avgpool1d -input $reshaped -output_size 2]
    
    # Should preserve batch and channel dims, reduce length
    set shape [torch::tensor_shape $result]
    string equal $shape {2 2 2}
} {1}

# Test 12: Single output size
test adaptive_avgpool1d-8.1 {Single output size} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {1 1 6}]
    set result [torch::adaptive_avgpool1d -input $reshaped -output_size 1]
    
    # Should reduce to single value
    set shape [torch::tensor_shape $result]
    string equal $shape {1 1 1}
} {1}

cleanupTests 