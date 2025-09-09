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
test conv_transpose3d-1.1 {Basic positional syntax with required parameters} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set result [torch::conv_transpose3d $input $weight]
    
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    
    list [llength $shape] $dtype
} {5 Float32}

test conv_transpose3d-1.2 {Positional syntax with bias} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set bias [torch::randn -shape {16}]
    set result [torch::conv_transpose3d $input $weight $bias]
    
    set shape [torch::tensor_shape $result]
    list [llength $shape] [lindex $shape 1]
} {5 16}

test conv_transpose3d-1.3 {Positional syntax with stride} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set result1 [torch::conv_transpose3d $input $weight none 1]
    set result2 [torch::conv_transpose3d $input $weight none 2]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    # Stride 2 should produce larger output
    list [lindex $shape1 2] [lindex $shape2 2]
} {6 9}

test conv_transpose3d-1.4 {Positional syntax with all parameters} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set bias [torch::randn -shape {16}]
    set result [torch::conv_transpose3d $input $weight $bias 2 1 1 1 1]
    
    set shape [torch::tensor_shape $result]
    set output_depth [lindex $shape 2]
    set output_height [lindex $shape 3]
    set output_width [lindex $shape 4]
    
    # Check dimensions are reasonable
    expr {$output_depth > 6 && $output_height > 6 && $output_width > 6}
} {1}

# Test 2: Named parameter syntax
test conv_transpose3d-2.1 {Named parameter syntax with required parameters} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set result [torch::conv_transpose3d -input $input -weight $weight]
    
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    
    list [llength $shape] $dtype
} {5 Float32}

test conv_transpose3d-2.2 {Named parameter syntax with bias} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set bias [torch::randn -shape {16}]
    set result [torch::conv_transpose3d -input $input -weight $weight -bias $bias]
    
    set shape [torch::tensor_shape $result]
    list [llength $shape] [lindex $shape 1]
} {5 16}

test conv_transpose3d-2.3 {Named parameter syntax with stride} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set result [torch::conv_transpose3d -input $input -weight $weight -stride 2]
    
    set shape [torch::tensor_shape $result]
    set output_depth [lindex $shape 2]
    
    # Stride 2 should produce larger output than default
    expr {$output_depth > 6}
} {1}

test conv_transpose3d-2.4 {Named parameter syntax with vector stride} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set result [torch::conv_transpose3d -input $input -weight $weight -stride {2 1 1}]
    
    set shape [torch::tensor_shape $result]
    set output_depth [lindex $shape 2]
    set output_height [lindex $shape 3]
    
    # Only depth dimension should be larger with stride {2 1 1}
    list [expr {$output_depth > 6}] [expr {$output_height == 6}]
} {1 1}

test conv_transpose3d-2.5 {Named parameter syntax with all parameters} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set bias [torch::randn -shape {16}]
    set result [torch::conv_transpose3d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -output_padding 1 -groups 1 -dilation 1]
    
    set shape [torch::tensor_shape $result]
    set output_channels [lindex $shape 1]
    
    list [llength $shape] $output_channels
} {5 16}

# Test 3: CamelCase alias
test conv_transpose3d-3.1 {CamelCase alias basic functionality} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set result [torch::convTranspose3d -input $input -weight $weight]
    
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    
    list [llength $shape] $dtype
} {5 Float32}

test conv_transpose3d-3.2 {CamelCase alias with outputPadding parameter} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set result [torch::convTranspose3d -input $input -weight $weight -stride 2 -outputPadding 1]
    
    set shape [torch::tensor_shape $result]
    set output_depth [lindex $shape 2]
    
    # Output padding should increase output size (stride=2, output_padding=1)
    expr {$output_depth > 8}
} {1}

test conv_transpose3d-3.3 {CamelCase alias functionality consistency} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    
    set result1 [torch::conv_transpose3d -input $input -weight $weight -stride 2]
    set result2 [torch::convTranspose3d -input $input -weight $weight -stride 2]
    
    tensors_equal $result1 $result2
} {1}

# Test 4: Error handling
test conv_transpose3d-4.1 {Error: Missing required input parameter} {
    set weight [torch::randn -shape {8 16 3 3 3}]
    catch {torch::conv_transpose3d -weight $weight} error
    string match "*input*" $error
} {1}

test conv_transpose3d-4.2 {Error: Missing required weight parameter} {
    set input [torch::randn -shape {2 8 4 4 4}]
    catch {torch::conv_transpose3d -input $input} error
    string match "*weight*" $error
} {1}

test conv_transpose3d-4.3 {Error: Invalid tensor name} {
    set weight [torch::randn -shape {8 16 3 3 3}]
    catch {torch::conv_transpose3d invalid_tensor $weight} error
    string match "*Invalid input tensor*" $error
} {1}

test conv_transpose3d-4.4 {Error: Invalid weight tensor name} {
    set input [torch::randn -shape {2 8 4 4 4}]
    catch {torch::conv_transpose3d $input invalid_weight} error
    string match "*Invalid weight tensor*" $error
} {1}

test conv_transpose3d-4.5 {Error: Invalid bias tensor name} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    catch {torch::conv_transpose3d -input $input -weight $weight -bias invalid_bias} error
    string match "*Invalid bias tensor*" $error
} {1}

test conv_transpose3d-4.6 {Error: Unknown parameter} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    catch {torch::conv_transpose3d -input $input -weight $weight -unknown_param 1} error
    string match "*Unknown parameter*" $error
} {1}

test conv_transpose3d-4.7 {Error: Missing parameter value} {
    set input [torch::randn -shape {2 8 4 4 4}]
    catch {torch::conv_transpose3d -input $input -weight} error
    string match "*pairs*" $error
} {1}

# Test 5: Mathematical correctness
test conv_transpose3d-5.1 {Output shape calculation - basic} {
    # Input: (N=1, C_in=2, D=3, H=3, W=3), Kernel: (C_in=2, C_out=4, kD=2, kH=2, kW=2)
    # Output should be: (N=1, C_out=4, D=4, H=4, W=4) with default parameters
    set input [torch::randn -shape {1 2 3 3 3}]
    set weight [torch::randn -shape {2 4 2 2 2}]
    set result [torch::conv_transpose3d $input $weight]
    
    set shape [torch::tensor_shape $result]
    list [lindex $shape 0] [lindex $shape 1] [lindex $shape 2] [lindex $shape 3] [lindex $shape 4]
} {1 4 4 4 4}

test conv_transpose3d-5.2 {Output shape calculation - with stride} {
    # With stride=2, output should be larger
    set input [torch::randn -shape {1 2 3 3 3}]
    set weight [torch::randn -shape {2 4 2 2 2}]
    set result [torch::conv_transpose3d $input $weight none 2]
    
    set shape [torch::tensor_shape $result]
    list [lindex $shape 2] [lindex $shape 3] [lindex $shape 4]
} {6 6 6}

test conv_transpose3d-5.3 {Output shape calculation - with padding} {
    # With padding=1, output should be smaller
    set input [torch::randn -shape {1 2 3 3 3}]
    set weight [torch::randn -shape {2 4 2 2 2}]
    set result [torch::conv_transpose3d $input $weight none 1 1]
    
    set shape [torch::tensor_shape $result]
    list [lindex $shape 2] [lindex $shape 3] [lindex $shape 4]
} {2 2 2}

test conv_transpose3d-5.4 {Output shape calculation - with output padding} {
    set input [torch::randn -shape {1 2 3 3 3}]
    set weight [torch::randn -shape {2 4 2 2 2}]
    set result1 [torch::conv_transpose3d $input $weight none 2 0 0]
    set result2 [torch::conv_transpose3d $input $weight none 2 0 1]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    # Output padding should increase output size (stride=2, output_padding=1)
    list [lindex $shape1 2] [lindex $shape2 2]
} {6 7}

test conv_transpose3d-5.5 {Volume processing example} {
    # Test with realistic volumetric data dimensions
    # Input: batch=2, channels=32, depth=8, height=16, width=16  
    # Weight: in_channels=32, out_channels=64, kernel=4x4x4
    set input [torch::randn -shape {2 32 8 16 16}]
    set weight [torch::randn -shape {32 64 4 4 4}]
    set bias [torch::randn -shape {64}]
    
    set result [torch::conv_transpose3d $input $weight $bias 2 1 0 1 1]
    
    set shape [torch::tensor_shape $result]
    set batch_size [lindex $shape 0]
    set out_channels [lindex $shape 1]
    
    list $batch_size $out_channels
} {2 64}

test conv_transpose3d-5.6 {Medical imaging example} {
    # Test with medical imaging dimensions (like CT scans)
    # Input: batch=1, channels=1, depth=64, height=64, width=64
    # Upsampling for super-resolution
    set input [torch::randn -shape {1 1 64 64 64}]
    set weight [torch::randn -shape {1 1 3 3 3}]
    
    set result [torch::conv_transpose3d $input $weight none 2 1 1]
    
    set shape [torch::tensor_shape $result]
    set output_depth [lindex $shape 2]
    set output_height [lindex $shape 3]
    set output_width [lindex $shape 4]
    
    # Should approximately double the spatial dimensions
    list [expr {$output_depth > 120}] [expr {$output_height > 120}] [expr {$output_width > 120}]
} {1 1 1}

test conv_transpose3d-5.7 {Groups parameter functionality} {
    # Test grouped transposed convolution
    set input [torch::randn -shape {1 4 3 3 3}]
    # 4 input channels, 2 output channels per group
    set weight [torch::randn -shape {4 2 2 2 2}]
    
    set result [torch::conv_transpose3d $input $weight none 1 0 0 2 1]
    
    set shape [torch::tensor_shape $result]
    set out_channels [lindex $shape 1]
    
    # With groups=2, we should get 2*2=4 output channels
    expr {$out_channels == 4}
} {1}

test conv_transpose3d-5.8 {Dilation parameter functionality} {
    # Test dilated transposed convolution
    set input [torch::randn -shape {1 2 4 4 4}]
    set weight [torch::randn -shape {2 4 3 3 3}]
    
    set result1 [torch::conv_transpose3d $input $weight none 1 0 0 1 1]
    set result2 [torch::conv_transpose3d $input $weight none 1 0 0 1 2]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    # Dilation should affect output size
    list [lindex $shape1 2] [lindex $shape2 2]
} {6 8}

# Test 6: Syntax consistency 
test conv_transpose3d-6.1 {Positional and named syntax produce same results} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    set bias [torch::randn -shape {16}]
    
    set result1 [torch::conv_transpose3d $input $weight $bias 2 1 1 1 1]
    set result2 [torch::conv_transpose3d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -output_padding 1 -groups 1 -dilation 1]
    
    tensors_equal $result1 $result2
} {1}

test conv_transpose3d-6.2 {Snake_case and camelCase produce same results} {
    set input [torch::randn -shape {2 8 4 4 4}]
    set weight [torch::randn -shape {8 16 3 3 3}]
    
    set result1 [torch::conv_transpose3d -input $input -weight $weight -stride 2 -output_padding 1]
    set result2 [torch::convTranspose3d -input $input -weight $weight -stride 2 -outputPadding 1]
    
    tensors_equal $result1 $result2
} {1}

cleanupTests 