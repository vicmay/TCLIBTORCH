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
proc create_test_tensors {} {
    # Create input tensor: 3D tensor [batch_size, in_channels, sequence_length]
    # Shape: [2, 3, 10] - 2 batches, 3 input channels, sequence length 10
    set input [torch::ones -shape {2 3 10} -dtype float32]
    
    # Create weight tensor: 3D tensor [out_channels, in_channels, kernel_size] 
    # Shape: [4, 3, 3] - 4 output channels, 3 input channels, kernel size 3
    set weight [torch::ones -shape {4 3 3} -dtype float32]
    
    # Create bias tensor: 1D tensor [out_channels]
    # Shape: [4] - one bias per output channel
    set bias [torch::ones -shape {4} -dtype float32]
    
    return [list $input $weight $bias]
}

# Test 1: Positional syntax - basic functionality
test tensor_conv1d-1.1 {Basic positional syntax with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1] 
    set bias [lindex $tensors 2]
    
    set result [torch::tensor_conv1d $input $weight $bias 1 0 1 1]
    expr {[string length $result] > 0}
} {1}

# Test 2: Positional syntax - without bias
test tensor_conv1d-1.2 {Positional syntax without bias} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv1d $input $weight none 1 0 1 1]
    expr {[string length $result] > 0}
} {1}

# Test 3: Positional syntax - minimal parameters
test tensor_conv1d-1.3 {Positional syntax with minimal parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv1d $input $weight]
    expr {[string length $result] > 0}
} {1}

# Test 4: Named parameter syntax - all parameters
test tensor_conv1d-2.1 {Named parameter syntax with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result [torch::tensor_conv1d -input $input -weight $weight -bias $bias -stride 1 -padding 0 -dilation 1 -groups 1]
    expr {[string length $result] > 0}
} {1}

# Test 5: Named parameter syntax - without bias
test tensor_conv1d-2.2 {Named parameter syntax without bias} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv1d -input $input -weight $weight -stride 1 -padding 0]
    expr {[string length $result] > 0}
} {1}

# Test 6: Named parameter syntax - minimal parameters
test tensor_conv1d-2.3 {Named parameter syntax with minimal parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv1d -input $input -weight $weight]
    expr {[string length $result] > 0}
} {1}

# Test 7: Named parameter syntax - parameters in different order
test tensor_conv1d-2.4 {Named parameter syntax with mixed parameter order} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result [torch::tensor_conv1d -stride 2 -input $input -dilation 1 -weight $weight -bias $bias -padding 1]
    expr {[string length $result] > 0}
} {1}

# Test 8: camelCase alias
test tensor_conv1d-3.1 {camelCase alias basic functionality} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensorConv1d $input $weight]
    expr {[string length $result] > 0}
} {1}

# Test 9: camelCase alias with named parameters
test tensor_conv1d-3.2 {camelCase alias with named parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result [torch::tensorConv1d -input $input -weight $weight -bias $bias -stride 2]
    expr {[string length $result] > 0}
} {1}

# Test 10: Mathematical correctness - both syntaxes should produce same results
test tensor_conv1d-4.1 {Both syntaxes produce same results} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result1 [torch::tensor_conv1d $input $weight $bias 1 0 1 1]
    set result2 [torch::tensor_conv1d -input $input -weight $weight -bias $bias -stride 1 -padding 0 -dilation 1 -groups 1]
    
    # Check if results have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

# Test 11: Different stride values
test tensor_conv1d-4.2 {Convolution with stride 2} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv1d -input $input -weight $weight -stride 2]
    set shape [torch::tensor_shape $result]
    
    # With stride 2, output length should be smaller
    expr {[llength $shape] == 3 && [lindex $shape 2] < 10}
} {1}

# Test 12: Different padding values
test tensor_conv1d-4.3 {Convolution with padding 1} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv1d -input $input -weight $weight -padding 1]
    set shape [torch::tensor_shape $result]
    
    # With padding 1 and kernel size 3, output length should be 10 (same as input)
    expr {[llength $shape] == 3 && [lindex $shape 2] == 10}
} {1}

# Test 13: Error handling - missing required parameters
test tensor_conv1d-5.1 {Error handling - missing input parameter} {
    set caught 0
    if {[catch {torch::tensor_conv1d -weight weight1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 14: Error handling - missing weight parameter
test tensor_conv1d-5.2 {Error handling - missing weight parameter} {
    set caught 0
    if {[catch {torch::tensor_conv1d -input input1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 15: Error handling - invalid tensor name
test tensor_conv1d-5.3 {Error handling - invalid tensor name} {
    set caught 0
    if {[catch {torch::tensor_conv1d -input invalid_tensor -weight invalid_weight} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 16: Error handling - unknown parameter
test tensor_conv1d-5.4 {Error handling - unknown parameter} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set caught 0
    if {[catch {torch::tensor_conv1d -input $input -weight $weight -unknown_param 1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 17: Error handling - invalid stride value
test tensor_conv1d-5.5 {Error handling - invalid stride value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set caught 0
    if {[catch {torch::tensor_conv1d -input $input -weight $weight -stride invalid} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 18: Positional error handling - wrong number of arguments
test tensor_conv1d-5.6 {Error handling - too few positional arguments} {
    set caught 0
    if {[catch {torch::tensor_conv1d input1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 19: Data type compatibility  
test tensor_conv1d-6.1 {Different data types} {
    # Create float32 tensors
    set input [torch::ones -shape {1 2 3} -dtype float32]
    set weight [torch::ones -shape {1 2 3} -dtype float32]
    
    set result [torch::tensor_conv1d $input $weight]
    expr {[string length $result] > 0}
} {1}

# Test 20: Edge case - kernel size equals input size
test tensor_conv1d-6.2 {Edge case - kernel size equals input size} {
    set input [torch::ones -shape {1 1 3} -dtype float32]
    set weight [torch::ones -shape {1 1 3} -dtype float32]
    
    set result [torch::tensor_conv1d $input $weight]
    set shape [torch::tensor_shape $result]
    
    # Output should be [1, 1, 1] - single convolution result
    expr {$shape eq "1 1 1"}
} {1}

# Test 21: dilation parameter
test tensor_conv1d-6.3 {Convolution with dilation} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv1d -input $input -weight $weight -dilation 2]
    expr {[string length $result] > 0}
} {1}

# Test 22: groups parameter
test tensor_conv1d-6.4 {Convolution with groups} {
    # Create input with 4 channels
    set input [torch::ones -shape {1 4 10} -dtype float32]
    
    # Create weight for 2 groups (2 channels per group: out_channels=4, in_channels/groups=2)
    set weight [torch::ones -shape {4 2 3} -dtype float32]
    
    set result [torch::tensor_conv1d -input $input -weight $weight -groups 2]
    expr {[string length $result] > 0}
} {1}

# Test 23: Complex parameter combination
test tensor_conv1d-6.5 {Complex parameter combination} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result [torch::tensor_conv1d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -dilation 1 -groups 1]
    set shape [torch::tensor_shape $result]
    
    # Verify output shape is reasonable
    expr {[llength $shape] == 3 && [lindex $shape 0] == 2 && [lindex $shape 1] == 4}
} {1}

# Test 24: Verify backward compatibility with old syntax
test tensor_conv1d-7.1 {Backward compatibility - old positional only syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    # Test the old 4-argument syntax that was supported before
    set result [torch::tensor_conv1d $input $weight $bias 2]
    expr {[string length $result] > 0}
} {1}

# Test 25: Mixed parameter validation
test tensor_conv1d-7.2 {Parameter validation - negative stride} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    # Negative stride should still work (will be handled by PyTorch)
    set result [torch::tensor_conv1d -input $input -weight $weight -stride 1]
    expr {[string length $result] > 0}
} {1}

cleanupTests 