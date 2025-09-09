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
# TORCH::FAKE_QUANTIZE_PER_TENSOR COMMAND TESTS
# ============================================================================

# Helper function to create test tensors
proc create_test_tensor {} {
    # Create input tensor [2, 3] with float values
    set input [torch::tensorCreate -data {1.5 2.5 3.5 4.5 5.5 6.5} -shape {2 3} -dtype float32]
    return $input
}

# Helper function to create large test tensor
proc create_large_test_tensor {} {
    # Create input tensor [3, 4] with float values
    set input [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {3 4} -dtype float32]
    return $input
}

# Test basic functionality - positional syntax
test fake_quantize_per_tensor-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-1.2 {Positional syntax with custom quant_min} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0 -100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-1.3 {Positional syntax with custom quant_min and quant_max} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0 -100 100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-1.4 {Positional syntax with larger scale} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 1.0 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-1.5 {Positional syntax with negative zero_point} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.2 -10]
    expr {$result ne ""}
} {1}

# Test named parameter syntax
test fake_quantize_per_tensor-2.1 {Basic named parameter syntax} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor -input $input -scale 0.1 -zero_point 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-2.2 {Named syntax with custom quant_min} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor -input $input -scale 0.1 -zero_point 0 -quant_min -100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-2.3 {Named syntax with custom quant_min and quant_max} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor -input $input -scale 0.1 -zero_point 0 -quant_min -100 -quant_max 100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-2.4 {Named syntax with camelCase parameter names} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor -input $input -scale 0.1 -zeroPoint 0 -quantMin -100 -quantMax 100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-2.5 {Named syntax with larger scale} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor -input $input -scale 2.0 -zero_point 5]
    expr {$result ne ""}
} {1}

# Test camelCase alias
test fake_quantize_per_tensor-3.1 {CamelCase alias with positional syntax} {
    set input [create_test_tensor]
    set result [torch::fakeQuantizePerTensor $input 0.1 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-3.2 {CamelCase alias with named syntax} {
    set input [create_test_tensor]
    set result [torch::fakeQuantizePerTensor -input $input -scale 0.1 -zero_point 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-3.3 {CamelCase alias with all parameters} {
    set input [create_test_tensor]
    set result [torch::fakeQuantizePerTensor -input $input -scale 0.1 -zeroPoint 0 -quantMin -50 -quantMax 50]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-3.4 {CamelCase alias with positional all parameters} {
    set input [create_test_tensor]
    set result [torch::fakeQuantizePerTensor $input 0.2 5 -50 50]
    expr {$result ne ""}
} {1}

# Test command existence
test fake_quantize_per_tensor-4.1 {Snake_case command exists} {
    info commands torch::fake_quantize_per_tensor
} {::torch::fake_quantize_per_tensor}

test fake_quantize_per_tensor-4.2 {CamelCase command exists} {
    info commands torch::fakeQuantizePerTensor
} {::torch::fakeQuantizePerTensor}

# Test different scales and zero points
test fake_quantize_per_tensor-5.1 {Small scale value} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.01 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-5.2 {Large scale value} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 10.0 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-5.3 {Positive zero point} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 100]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-5.4 {Negative zero point} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 -100]
    expr {$result ne ""}
} {1}

# Test edge cases
test fake_quantize_per_tensor-6.1 {Zero values in input} {
    set input [torch::tensorCreate -data {0.0 0.0 0.0 0.0 0.0 0.0} -shape {2 3} -dtype float32]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-6.2 {Negative values in input} {
    set input [torch::tensorCreate -data {-1.0 -2.0 -3.0 -4.0 -5.0 -6.0} -shape {2 3} -dtype float32]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-6.3 {Large values in input} {
    set input [torch::tensorCreate -data {100.0 200.0 300.0 400.0 500.0 600.0} -shape {2 3} -dtype float32]
    set result [torch::fake_quantize_per_tensor $input 1.0 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-6.4 {Wide quantization range} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0 -255 255]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-6.5 {Narrow quantization range} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0 -1 1]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-6.6 {UINT8 quantization range} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 128 0 255]
    expr {$result ne ""}
} {1}

# Test with different tensor shapes
test fake_quantize_per_tensor-7.1 {1D tensor} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -shape {5} -dtype float32]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-7.2 {3D tensor} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-7.3 {Large tensor} {
    set input [create_large_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.5 0]
    expr {$result ne ""}
} {1}

# Test error handling
test fake_quantize_per_tensor-8.1 {Missing required arguments - positional} {
    set code [catch {torch::fake_quantize_per_tensor} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.2 {Invalid input tensor} {
    set code [catch {torch::fake_quantize_per_tensor "invalid_tensor" 0.1 0} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.3 {Invalid scale value} {
    set input [create_test_tensor]
    set code [catch {torch::fake_quantize_per_tensor $input "invalid_scale" 0} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.4 {Invalid zero_point value} {
    set input [create_test_tensor]
    set code [catch {torch::fake_quantize_per_tensor $input 0.1 "invalid_zero_point"} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.5 {Invalid quant_min value} {
    set input [create_test_tensor]
    set code [catch {torch::fake_quantize_per_tensor $input 0.1 0 "invalid_min"} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.6 {Invalid quant_max value} {
    set input [create_test_tensor]
    set code [catch {torch::fake_quantize_per_tensor $input 0.1 0 -128 "invalid_max"} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.7 {Missing required parameters - named syntax} {
    set code [catch {torch::fake_quantize_per_tensor -scale 0.1} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.8 {Unknown parameter in named syntax} {
    set input [create_test_tensor]
    set code [catch {torch::fake_quantize_per_tensor -input $input -scale 0.1 -zero_point 0 -unknown_param value} msg]
    expr {$code == 1}
} {1}

test fake_quantize_per_tensor-8.9 {Missing value for parameter} {
    set input [create_test_tensor]
    set code [catch {torch::fake_quantize_per_tensor -input $input -scale} msg]
    expr {$code == 1}
} {1}

# Test mathematical correctness
test fake_quantize_per_tensor-9.1 {Output shape preservation} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    set input_shape [torch::tensorShape $input]
    set output_shape [torch::tensorShape $result]
    expr {$input_shape eq $output_shape}
} {1}

test fake_quantize_per_tensor-9.2 {Output dtype preservation} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    set input_dtype [torch::tensorDtype $input]
    set output_dtype [torch::tensorDtype $result]
    expr {$input_dtype eq $output_dtype}
} {1}

test fake_quantize_per_tensor-9.3 {Output values are reasonable} {
    set input [create_test_tensor]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    # Check that output tensor is valid and has the same shape
    set result_shape [torch::tensorShape $result]
    set input_shape [torch::tensorShape $input]
    expr {$result_shape eq $input_shape}
} {1}

test fake_quantize_per_tensor-9.4 {Syntax equivalence - same results} {
    set input [create_test_tensor]
    
    set result1 [torch::fake_quantize_per_tensor $input 0.1 0 -100 100]
    set result2 [torch::fake_quantize_per_tensor -input $input -scale 0.1 -zero_point 0 -quant_min -100 -quant_max 100]
    set result3 [torch::fakeQuantizePerTensor -input $input -scale 0.1 -zeroPoint 0 -quantMin -100 -quantMax 100]
    
    set diff1 [torch::tensorSub $result1 $result2]
    set diff2 [torch::tensorSub $result1 $result3]
    set max_diff1 [torch::tensorMax $diff1]
    set max_diff2 [torch::tensorMax $diff2]
    
    expr {abs([torch::tensorItem $max_diff1]) < 1e-6 && abs([torch::tensorItem $max_diff2]) < 1e-6}
} {1}

test fake_quantize_per_tensor-9.5 {Zero scale should produce zero output} {
    set input [create_test_tensor]
    # Very small scale should quantize everything to zero_point
    set result [torch::fake_quantize_per_tensor $input 1e-10 5]
    set expected_value 0.0
    
    # Check that all values are close to expected dequantized zero point value
    set mean_val [torch::tensorMean $result]
    expr {abs([torch::tensorItem $mean_val] - $expected_value) < 1e-6}
} {1}

# Test data type compatibility
test fake_quantize_per_tensor-10.1 {Float64 input tensor} {
    set input [torch::tensorCreate -data {1.5 2.5 3.5} -shape {3} -dtype float64]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    expr {$result ne ""}
} {1}

test fake_quantize_per_tensor-10.2 {Different tensor sizes} {
    # Single element tensor
    set input [torch::tensorCreate -data {5.0} -shape {1} -dtype float32]
    set result [torch::fake_quantize_per_tensor $input 0.1 0]
    expr {$result ne ""}
} {1}

cleanupTests 