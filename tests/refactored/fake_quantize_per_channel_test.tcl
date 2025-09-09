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

# ============================================================================
# TORCH::FAKE_QUANTIZE_PER_CHANNEL COMMAND TESTS
# ============================================================================

# Helper function to create test tensors
proc create_test_tensors {} {
    # Create input tensor [2, 3] with float values
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    
    # Create scales tensor [3] (per-channel scales)
    set scales [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]
    
    # Create zero_points tensor [3] (per-channel zero points)
    set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
    
    return [list $input $scales $zero_points]
}

# Helper function to create larger test tensors
proc create_large_test_tensors {} {
    # Create input tensor [2, 4, 3] with float values
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0} -shape {2 4 3} -dtype float32]
    
    # Create scales tensor [4] (per-channel scales for axis 1)
    set scales [torch::tensorCreate -data {0.1 0.2 0.3 0.4} -shape {4} -dtype float32]
    
    # Create zero_points tensor [4] (per-channel zero points for axis 1)
    set zero_points [torch::tensorCreate -data {0 1 2 3} -shape {4} -dtype int32]
    
    return [list $input $scales $zero_points]
}

# Test basic functionality - positional syntax
test fake_quantize_per_channel-1.1 {Basic positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-1.2 {Positional syntax with custom quant_min} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1 -100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-1.3 {Positional syntax with custom quant_min and quant_max} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1 -100 100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-1.4 {Positional syntax with axis 0} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [torch::tensorCreate -data {0.1 0.2} -shape {2} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1} -shape {2} -dtype int32]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 0]
    expr {$result ne ""}
} {1}

# Test named parameter syntax
test fake_quantize_per_channel-2.1 {Basic named parameter syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel -input $input -scales $scales -zero_points $zero_points -axis 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-2.2 {Named syntax with custom quant_min} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel -input $input -scales $scales -zero_points $zero_points -axis 1 -quant_min -100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-2.3 {Named syntax with custom quant_min and quant_max} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel -input $input -scales $scales -zero_points $zero_points -axis 1 -quant_min -100 -quant_max 100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-2.4 {Named syntax with camelCase parameter names} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel -input $input -scales $scales -zeroPoints $zero_points -axis 1 -quantMin -100 -quantMax 100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-2.5 {Named syntax with axis 0} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [torch::tensorCreate -data {0.1 0.2} -shape {2} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1} -shape {2} -dtype int32]
    
    set result [torch::fake_quantize_per_channel -input $input -scales $scales -zero_points $zero_points -axis 0]
    expr {$result ne ""}
} {1}

# Test camelCase alias
test fake_quantize_per_channel-3.1 {CamelCase alias with positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fakeQuantizePerChannel $input $scales $zero_points 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-3.2 {CamelCase alias with named syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fakeQuantizePerChannel -input $input -scales $scales -zero_points $zero_points -axis 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-3.3 {CamelCase alias with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fakeQuantizePerChannel -input $input -scales $scales -zeroPoints $zero_points -axis 1 -quantMin -50 -quantMax 50]
    expr {$result ne ""}
} {1}

# Test command existence
test fake_quantize_per_channel-4.1 {Snake_case command exists} {
    info commands torch::fake_quantize_per_channel
} {::torch::fake_quantize_per_channel}

test fake_quantize_per_channel-4.2 {CamelCase command exists} {
    info commands torch::fakeQuantizePerChannel
} {::torch::fakeQuantizePerChannel}

# Test different axes with larger tensors
test fake_quantize_per_channel-5.1 {3D tensor with axis 1} {
    set tensors [create_large_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-5.2 {3D tensor with axis 2} {
    set tensors [create_large_test_tensors]
    set input [lindex $tensors 0]
    set scales [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 2]
    expr {$result ne ""}
} {1}

# Test edge cases
test fake_quantize_per_channel-6.1 {Zero values in input} {
    set input [torch::tensorCreate -data {0.0 0.0 0.0 0.0 0.0 0.0} -shape {2 3} -dtype float32]
    set scales [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-6.2 {Negative values in input} {
    set input [torch::tensorCreate -data {-1.0 -2.0 -3.0 -4.0 -5.0 -6.0} -shape {2 3} -dtype float32]
    set scales [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-6.3 {Large values in input} {
    set input [torch::tensorCreate -data {100.0 200.0 300.0 400.0 500.0 600.0} -shape {2 3} -dtype float32]
    set scales [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-6.4 {Wide quantization range} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1 -255 255]
    expr {$result ne ""}
} {1}

test fake_quantize_per_channel-6.5 {Narrow quantization range} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    # Use zero_points within the range [-1, 1]
    set zero_points [torch::tensorCreate -data {-1 0 1} -shape {3} -dtype int32]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1 -1 1]
    expr {$result ne ""}
} {1}

# Test error handling
test fake_quantize_per_channel-7.1 {Missing required arguments - positional} {
    set code [catch {torch::fake_quantize_per_channel} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.2 {Invalid input tensor} {
    set scales [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
    
    set code [catch {torch::fake_quantize_per_channel "invalid_tensor" $scales $zero_points 1} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.3 {Invalid scales tensor} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
    
    set code [catch {torch::fake_quantize_per_channel $input "invalid_tensor" $zero_points 1} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.4 {Invalid zero_points tensor} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set scales [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]
    
    set code [catch {torch::fake_quantize_per_channel $input $scales "invalid_tensor" 1} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.5 {Invalid axis value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set code [catch {torch::fake_quantize_per_channel $input $scales $zero_points "invalid_axis"} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.6 {Invalid quant_min value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set code [catch {torch::fake_quantize_per_channel $input $scales $zero_points 1 "invalid_min"} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.7 {Invalid quant_max value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set code [catch {torch::fake_quantize_per_channel $input $scales $zero_points 1 -128 "invalid_max"} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.8 {Missing required parameters - named syntax} {
    set code [catch {torch::fake_quantize_per_channel -axis 1} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_channel-7.9 {Unknown parameter in named syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set code [catch {torch::fake_quantize_per_channel -input $input -scales $scales -zero_points $zero_points -axis 1 -unknown_param value} msg]
    expr {$code == 1}
} {1}

# Test mathematical correctness
test fake_quantize_per_channel-8.1 {Output shape preservation} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    set input_shape [torch::tensorShape $input]
    set output_shape [torch::tensorShape $result]
    expr {$input_shape eq $output_shape}
} {1}

test fake_quantize_per_channel-8.2 {Output dtype preservation} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    set input_dtype [torch::tensorDtype $input]
    set output_dtype [torch::tensorDtype $result]
    expr {$input_dtype eq $output_dtype}
} {1}

test fake_quantize_per_channel-8.3 {Output values are reasonable} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result [torch::fake_quantize_per_channel $input $scales $zero_points 1]
    # Check that output tensor is valid and has the same shape
    set result_shape [torch::tensorShape $result]
    set input_shape [torch::tensorShape $input]
    expr {$result_shape eq $input_shape}
} {1}

test fake_quantize_per_channel-8.4 {Syntax equivalence - same results} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set scales [lindex $tensors 1]
    set zero_points [lindex $tensors 2]
    
    set result1 [torch::fake_quantize_per_channel $input $scales $zero_points 1 -100 100]
    set result2 [torch::fake_quantize_per_channel -input $input -scales $scales -zero_points $zero_points -axis 1 -quant_min -100 -quant_max 100]
    set result3 [torch::fakeQuantizePerChannel -input $input -scales $scales -zeroPoints $zero_points -axis 1 -quantMin -100 -quantMax 100]
    
    set diff1 [torch::tensorSub $result1 $result2]
    set diff2 [torch::tensorSub $result1 $result3]
    set max_diff1 [torch::tensorMax $diff1]
    set max_diff2 [torch::tensorMax $diff2]
    
    expr {abs([torch::tensorItem $max_diff1]) < 1e-6 && abs([torch::tensorItem $max_diff2]) < 1e-6}
} {1}

cleanupTests
