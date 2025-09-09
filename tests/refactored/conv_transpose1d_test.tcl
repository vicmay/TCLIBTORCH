#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Configure test output
configure -verbose {pass fail skip error}

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Helper function to check if two tensors are approximately equal
proc tensors_equal {tensor1 tensor2 {tolerance 1e-5}} {
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# Test 1: Basic positional syntax
test conv_transpose1d-1.1 {Basic positional syntax with required parameters} {
    set input [torch::randn -shape {1 1 5}]
    set weight [torch::randn -shape {1 1 2}]
    set result [torch::conv_transpose1d $input $weight]
    
    # Check result shape and basic properties
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    list [llength $shape] $dtype
} {3 Float32}

# Test 2: Positional syntax with bias
test conv_transpose1d-1.2 {Positional syntax with bias} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set bias [torch::randn -shape {1}]
    set result [torch::conv_transpose1d $input $weight $bias]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 3: Positional syntax with stride
test conv_transpose1d-1.3 {Positional syntax with stride} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set result [torch::conv_transpose1d $input $weight none 2]
    
    # Check result shape (stride=2 should increase output size)
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 4: Positional syntax with all parameters
test conv_transpose1d-1.4 {Positional syntax with all parameters} {
    set input [torch::randn -shape {1 1 4}]
    set weight [torch::randn -shape {1 1 2}]
    set bias [torch::randn -shape {1}]
    set result [torch::conv_transpose1d $input $weight $bias 1 0 0 1 1]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 5: Named parameter syntax - basic
test conv_transpose1d-2.1 {Named parameter syntax with required parameters} {
    set input [torch::randn -shape {1 1 5}]
    set weight [torch::randn -shape {1 1 2}]
    set result [torch::conv_transpose1d -input $input -weight $weight]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 6: Named parameter syntax with bias
test conv_transpose1d-2.2 {Named parameter syntax with bias} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set bias [torch::randn -shape {1}]
    set result [torch::conv_transpose1d -input $input -weight $weight -bias $bias]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 7: Named parameter syntax with stride
test conv_transpose1d-2.3 {Named parameter syntax with stride} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set result [torch::conv_transpose1d -input $input -weight $weight -stride 2]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 8: Named parameter syntax with all parameters
test conv_transpose1d-2.4 {Named parameter syntax with all parameters} {
    set input [torch::randn -shape {1 1 4}]
    set weight [torch::randn -shape {1 1 2}]
    set bias [torch::randn -shape {1}]
    set result [torch::conv_transpose1d -input $input -weight $weight -bias $bias -stride 1 -padding 0 -output_padding 0 -groups 1 -dilation 1]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 9: Named parameter syntax with mixed order
test conv_transpose1d-2.5 {Named parameter syntax with mixed parameter order} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set bias [torch::randn -shape {1}]
    set result [torch::conv_transpose1d -stride 1 -input $input -bias $bias -weight $weight]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 10: CamelCase alias
test conv_transpose1d-3.1 {CamelCase alias - basic usage} {
    set input [torch::randn -shape {1 1 5}]
    set weight [torch::randn -shape {1 1 2}]
    set result [torch::convTranspose1d $input $weight]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 11: CamelCase alias with named parameters
test conv_transpose1d-3.2 {CamelCase alias with named parameters} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set bias [torch::randn -shape {1}]
    set result [torch::convTranspose1d -input $input -weight $weight -bias $bias -stride 2]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 12: Syntax consistency - both syntaxes should produce same results
test conv_transpose1d-3.3 {Syntax consistency check} {
    set input [torch::ones -shape {1 1 4}]
    set weight [torch::ones -shape {1 1 2}]
    set bias [torch::zeros -shape {1}]
    
    # Positional syntax
    set result1 [torch::conv_transpose1d $input $weight $bias 2 1 0 1 1]
    
    # Named syntax
    set result2 [torch::conv_transpose1d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -output_padding 0 -groups 1 -dilation 1]
    
    # CamelCase syntax
    set result3 [torch::convTranspose1d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -output_padding 0 -groups 1 -dilation 1]
    
    # Check all results are equal
    set equal12 [tensors_equal $result1 $result2]
    set equal13 [tensors_equal $result1 $result3]
    
    list $equal12 $equal13
} {1 1}

# Test 13: Error handling - missing required parameters
test conv_transpose1d-4.1 {Error handling - missing input parameter} {
    set weight [torch::randn -shape {1 1 2}]
    set result [catch {torch::conv_transpose1d -weight $weight} msg]
    list $result [string match "*input*" $msg]
} {1 1}

# Test 14: Error handling - missing weight parameter
test conv_transpose1d-4.2 {Error handling - missing weight parameter} {
    set input [torch::randn -shape {1 1 3}]
    set result [catch {torch::conv_transpose1d -input $input} msg]
    list $result [string match "*weight*" $msg]
} {1 1}

# Test 15: Error handling - invalid tensor name
test conv_transpose1d-4.3 {Error handling - invalid tensor name} {
    set input [torch::randn -shape {1 1 3}]
    set result [catch {torch::conv_transpose1d $input invalid_tensor} msg]
    list $result [string match "*Invalid*tensor*" $msg]
} {1 1}

# Test 16: Error handling - invalid bias tensor
test conv_transpose1d-4.4 {Error handling - invalid bias tensor} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set result [catch {torch::conv_transpose1d -input $input -weight $weight -bias invalid_bias} msg]
    list $result [string match "*Invalid*bias*" $msg]
} {1 1}

# Test 17: Error handling - invalid stride value
test conv_transpose1d-4.5 {Error handling - invalid stride value} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set result [catch {torch::conv_transpose1d -input $input -weight $weight -stride invalid} msg]
    list $result [string match "*stride*" $msg]
} {1 1}

# Test 18: Error handling - unknown parameter
test conv_transpose1d-4.6 {Error handling - unknown parameter} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set result [catch {torch::conv_transpose1d -input $input -weight $weight -unknown_param 1} msg]
    list $result [string match "*Unknown parameter*" $msg]
} {1 1}

# Test 19: Error handling - unpaired parameters
test conv_transpose1d-4.7 {Error handling - unpaired parameters} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    set result [catch {torch::conv_transpose1d -input $input -weight $weight -stride} msg]
    list $result [string match "*pairs*" $msg]
} {1 1}

# Test 20: Mathematical correctness - simple case
test conv_transpose1d-5.1 {Mathematical correctness - simple upsampling} {
    # Create a simple input
    set input [torch::ones -shape {1 1 2}]
    # Create a simple kernel
    set weight [torch::ones -shape {1 1 1}]
    # Apply with stride=2 (should double the size)
    set result [torch::conv_transpose1d $input $weight none 2]
    
    # Check output shape and verify upsampling
    set shape [torch::tensor_shape $result]
    lindex $shape 2
} {3}

# Test 21: Multi-channel input
test conv_transpose1d-5.2 {Multi-channel input} {
    # Create 2-channel input
    set input [torch::randn -shape {1 2 3}]
    # Create appropriate weight
    set weight [torch::randn -shape {2 2 1}]
    set result [torch::conv_transpose1d $input $weight]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 22: Padding effect
test conv_transpose1d-5.3 {Padding effect} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    
    # Without padding
    set result1 [torch::conv_transpose1d $input $weight none 1 0]
    set shape1 [torch::tensor_shape $result1]
    
    # With padding
    set result2 [torch::conv_transpose1d $input $weight none 1 1]
    set shape2 [torch::tensor_shape $result2]
    
    # Padding should reduce output size
    list [lindex $shape1 2] [lindex $shape2 2]
} {4 2}

# Test 23: Different stride values
test conv_transpose1d-5.4 {Different stride values} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    
    # Stride 1
    set result1 [torch::conv_transpose1d $input $weight none 1]
    set shape1 [torch::tensor_shape $result1]
    
    # Stride 2 
    set result2 [torch::conv_transpose1d $input $weight none 2]
    set shape2 [torch::tensor_shape $result2]
    
    # Stride 2 should produce larger output
    list [lindex $shape1 2] [lindex $shape2 2]
} {4 6}

# Test 24: Groups parameter
test conv_transpose1d-5.5 {Groups parameter} {
    # Create 2-channel input for grouped convolution
    set input [torch::randn -shape {1 2 3}]
    # Create weight for 2 groups
    set weight [torch::randn -shape {2 1 1}]
    set result [torch::conv_transpose1d $input $weight none 1 0 0 2]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    llength $shape
} {3}

# Test 25: Dilation parameter
test conv_transpose1d-5.6 {Dilation parameter} {
    set input [torch::randn -shape {1 1 5}]
    set weight [torch::randn -shape {1 1 2}]
    
    # Without dilation
    set result1 [torch::conv_transpose1d $input $weight none 1 0 0 1 1]
    set shape1 [torch::tensor_shape $result1]
    
    # With dilation
    set result2 [torch::conv_transpose1d $input $weight none 1 0 0 1 2]
    set shape2 [torch::tensor_shape $result2]
    
    # Dilation should increase output size
    list [lindex $shape1 2] [lindex $shape2 2]
} {6 7}

# Test 26: Output padding
test conv_transpose1d-5.7 {Output padding parameter} {
    set input [torch::randn -shape {1 1 3}]
    set weight [torch::randn -shape {1 1 2}]
    
    # Without output padding
    set result1 [torch::conv_transpose1d $input $weight none 2 0 0]
    set shape1 [torch::tensor_shape $result1]
    
    # With output padding
    set result2 [torch::conv_transpose1d $input $weight none 2 0 1]
    set shape2 [torch::tensor_shape $result2]
    
    # Output padding should increase output size
    list [lindex $shape1 2] [lindex $shape2 2]
} {6 7}

# Test 27: Large tensor processing
test conv_transpose1d-5.8 {Large tensor processing} {
    # Create larger tensors
    set input [torch::randn -shape {1 1 100}]
    set weight [torch::randn -shape {1 1 3}]
    
    set result [torch::conv_transpose1d $input $weight]
    
    # Check result shape
    set shape [torch::tensor_shape $result]
    lindex $shape 2
} {102}

puts "All conv_transpose1d tests completed!"
cleanupTests 