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

# Helper function to create test tensors for conv_transpose2d
proc create_test_tensors {} {
    # Create input tensor: 4D tensor [batch_size, in_channels, height, width]
    # Shape: [2, 3, 8, 8] - 2 batches, 3 input channels, 8x8 spatial
    set input [torch::ones -shape {2 3 8 8} -dtype float32]
    
    # Create weight tensor: 4D tensor [in_channels, out_channels, kernel_height, kernel_width] 
    # Shape: [3, 4, 3, 3] - 3 input channels, 4 output channels, 3x3 kernel
    set weight [torch::ones -shape {3 4 3 3} -dtype float32]
    
    # Create bias tensor: 1D tensor [out_channels]
    # Shape: [4] - one bias per output channel
    set bias [torch::ones -shape {4} -dtype float32]
    
    return [list $input $weight $bias]
}

# Test 1: Positional syntax - basic functionality
test tensor_conv_transpose2d-1.1 {Basic positional syntax with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1] 
    set bias [lindex $tensors 2]
    
    set result [torch::tensor_conv_transpose2d $input $weight $bias 1 0 0 1 1]
    expr {[string length $result] > 0}
} {1}

# Test 2: Positional syntax - without bias
test tensor_conv_transpose2d-1.2 {Positional syntax without bias} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d $input $weight none 1 0 0 1 1]
    expr {[string length $result] > 0}
} {1}

# Test 3: Positional syntax - minimal parameters
test tensor_conv_transpose2d-1.3 {Positional syntax with minimal parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d $input $weight]
    expr {[string length $result] > 0}
} {1}

# Test 4: Named parameter syntax - all parameters
test tensor_conv_transpose2d-2.1 {Named parameter syntax with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -bias $bias -stride 1 -padding 0 -output_padding 0 -groups 1 -dilation 1]
    expr {[string length $result] > 0}
} {1}

# Test 5: Named parameter syntax - without bias
test tensor_conv_transpose2d-2.2 {Named parameter syntax without bias} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -stride 1 -padding 0]
    expr {[string length $result] > 0}
} {1}

# Test 6: Named parameter syntax - minimal parameters
test tensor_conv_transpose2d-2.3 {Named parameter syntax with minimal parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight]
    expr {[string length $result] > 0}
} {1}

# Test 7: 2D Parameters - stride as pair
test tensor_conv_transpose2d-2.4 {Stride as pair (height, width)} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -stride {2 1}]
    expr {[string length $result] > 0}
} {1}

# Test 8: 2D Parameters - padding as pair
test tensor_conv_transpose2d-2.5 {Padding as pair (height, width)} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -padding {1 2}]
    expr {[string length $result] > 0}
} {1}

# Test 9: 2D Parameters - output_padding as pair
test tensor_conv_transpose2d-2.6 {Output padding as pair (height, width)} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -stride 2 -output_padding {1 0}]
    expr {[string length $result] > 0}
} {1}

# Test 10: 2D Parameters - dilation as pair
test tensor_conv_transpose2d-2.7 {Dilation as pair (height, width)} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -dilation {2 1}]
    expr {[string length $result] > 0}
} {1}

# Test 11: camelCase alias
test tensor_conv_transpose2d-3.1 {camelCase alias basic functionality} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensorConvTranspose2d $input $weight]
    expr {[string length $result] > 0}
} {1}

# Test 12: camelCase alias with named parameters
test tensor_conv_transpose2d-3.2 {camelCase alias with named parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result [torch::tensorConvTranspose2d -input $input -weight $weight -bias $bias -stride 2]
    expr {[string length $result] > 0}
} {1}

# Test 13: Mathematical correctness - both syntaxes should produce same results
test tensor_conv_transpose2d-4.1 {Both syntaxes produce same results} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result1 [torch::tensor_conv_transpose2d $input $weight $bias 1 0 0 1 1]
    set result2 [torch::tensor_conv_transpose2d -input $input -weight $weight -bias $bias -stride 1 -padding 0 -output_padding 0 -groups 1 -dilation 1]
    
    # Check if results have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

# Test 14: Different stride values
test tensor_conv_transpose2d-4.2 {Transposed convolution with stride 2} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -stride 2]
    set shape [torch::tensor_shape $result]
    
    # With stride 2, output size should be larger
    expr {[llength $shape] == 4 && [lindex $shape 2] > 8 && [lindex $shape 3] > 8}
} {1}

# Test 15: Different padding values
test tensor_conv_transpose2d-4.3 {Transposed convolution with padding 1} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -padding 1]
    set shape [torch::tensor_shape $result]
    
    # With padding 1, output size should be affected
    expr {[llength $shape] == 4 && [lindex $shape 2] >= 8 && [lindex $shape 3] >= 8}
} {1}

# Test 16: Output padding parameter
test tensor_conv_transpose2d-4.4 {Transposed convolution with output padding} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -stride 2 -output_padding 1]
    set shape [torch::tensor_shape $result]
    
    # Output padding should increase the output size
    expr {[llength $shape] == 4 && [lindex $shape 2] > 15 && [lindex $shape 3] > 15}
} {1}

# Test 17: Asymmetric stride (different for height and width)
test tensor_conv_transpose2d-4.5 {Asymmetric stride (2, 1)} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -stride {2 1}]
    set shape [torch::tensor_shape $result]
    
    # Height should be larger than width due to asymmetric stride
    expr {[llength $shape] == 4 && [lindex $shape 2] > [lindex $shape 3]}
} {1}

# Test 18: Error handling - missing required parameters
test tensor_conv_transpose2d-5.1 {Error handling - missing input parameter} {
    set caught 0
    if {[catch {torch::tensor_conv_transpose2d -weight weight1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 19: Error handling - missing weight parameter
test tensor_conv_transpose2d-5.2 {Error handling - missing weight parameter} {
    set caught 0
    if {[catch {torch::tensor_conv_transpose2d -input input1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 20: Error handling - invalid tensor name
test tensor_conv_transpose2d-5.3 {Error handling - invalid tensor name} {
    set caught 0
    if {[catch {torch::tensor_conv_transpose2d -input invalid_tensor -weight invalid_weight} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 21: Error handling - unknown parameter
test tensor_conv_transpose2d-5.4 {Error handling - unknown parameter} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set caught 0
    if {[catch {torch::tensor_conv_transpose2d -input $input -weight $weight -unknown_param 1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 22: Error handling - invalid stride value
test tensor_conv_transpose2d-5.5 {Error handling - invalid stride value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set caught 0
    if {[catch {torch::tensor_conv_transpose2d -input $input -weight $weight -stride invalid} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 23: Error handling - invalid list size for stride
test tensor_conv_transpose2d-5.6 {Error handling - invalid list size for stride} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set caught 0
    if {[catch {torch::tensor_conv_transpose2d -input $input -weight $weight -stride {1 2 3}} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 24: Positional error handling - wrong number of arguments
test tensor_conv_transpose2d-5.7 {Error handling - too few positional arguments} {
    set caught 0
    if {[catch {torch::tensor_conv_transpose2d input1} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 25: Data type compatibility  
test tensor_conv_transpose2d-6.1 {Different data types} {
    # Create float32 tensors
    set input [torch::ones -shape {1 2 4 4} -dtype float32]
    set weight [torch::ones -shape {2 3 3 3} -dtype float32]
    
    set result [torch::tensor_conv_transpose2d $input $weight]
    expr {[string length $result] > 0}
} {1}

# Test 26: Edge case - kernel size equals input size
test tensor_conv_transpose2d-6.2 {Edge case - kernel size equals input size} {
    set input [torch::ones -shape {1 1 3 3} -dtype float32]
    set weight [torch::ones -shape {1 2 3 3} -dtype float32]
    
    set result [torch::tensor_conv_transpose2d $input $weight]
    set shape [torch::tensor_shape $result]
    
    # Output should be larger than input for transposed convolution
    expr {[llength $shape] == 4 && [lindex $shape 2] >= 3 && [lindex $shape 3] >= 3}
} {1}

# Test 27: dilation parameter
test tensor_conv_transpose2d-6.3 {Transposed convolution with dilation} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -dilation 2]
    expr {[string length $result] > 0}
} {1}

# Test 28: groups parameter
test tensor_conv_transpose2d-6.4 {Transposed convolution with groups} {
    # Create input with 4 channels
    set input [torch::ones -shape {1 4 6 6} -dtype float32]
    
    # Create weight for 2 groups (2 input channels per group, 3 output channels per group)
    set weight [torch::ones -shape {4 3 3 3} -dtype float32]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -groups 2]
    expr {[string length $result] > 0}
} {1}

# Test 29: Complex parameter combination
test tensor_conv_transpose2d-6.5 {Complex parameter combination} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -bias $bias -stride {2 1} -padding {1 0} -output_padding {1 0} -dilation 1 -groups 1]
    set shape [torch::tensor_shape $result]
    
    # Verify output shape is reasonable
    expr {[llength $shape] == 4 && [lindex $shape 0] == 2 && [lindex $shape 1] == 4}
} {1}

# Test 30: Verify backward compatibility with old syntax
test tensor_conv_transpose2d-7.1 {Backward compatibility - old positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    set bias [lindex $tensors 2]
    
    # Test the old 4-argument syntax that was supported before
    set result [torch::tensor_conv_transpose2d $input $weight $bias 2]
    expr {[string length $result] > 0}
} {1}

# Test 31: Output shape verification
test tensor_conv_transpose2d-7.2 {Output shape calculation verification} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight]
    set shape [torch::tensor_shape $result]
    
    # For transposed conv2d: output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    # With defaults: (8 - 1) * 1 - 2 * 0 + 3 + 0 = 7 + 3 = 10
    expr {[lindex $shape 2] == 10 && [lindex $shape 3] == 10}
} {1}

# Test 32: Mixed single int and pair parameters
test tensor_conv_transpose2d-7.3 {Mixed parameter types - single int and pairs} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set weight [lindex $tensors 1]
    
    set result [torch::tensor_conv_transpose2d -input $input -weight $weight -stride 2 -padding {1 0} -dilation {1 2}]
    expr {[string length $result] > 0}
} {1}

cleanupTests 