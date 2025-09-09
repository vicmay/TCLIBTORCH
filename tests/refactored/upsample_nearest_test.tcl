#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test cases for positional syntax
test upsample-nearest-1.1 {Basic positional syntax with 3D tensor} {
    ;# Create 3D input tensor (N=1, C=1, L=4) for 1D upsampling
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling with single size value for 1D spatial dimension
    set result [torch::upsample_nearest $input {8}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-1.2 {Positional syntax with 4D tensor} {
    ;# Create 4D input tensor (N=1, C=1, H=2, W=2) for 2D upsampling
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 2 2}]
    
    ;# Test upsampling to larger size (2 values for H,W)
    set result [torch::upsample_nearest $input {4 4}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-1.3 {Positional syntax with 5D tensor} {
    ;# Create 5D input tensor (N=1, C=1, D=2, H=2, W=2) for 3D upsampling
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 2 2 2}]
    
    ;# Test upsampling to larger size (3 values for D,H,W)
    set result [torch::upsample_nearest $input {4 4 4}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Test cases for named parameter syntax
test upsample-nearest-2.1 {Named parameter syntax with -input and -size} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling with named parameters
    set result [torch::upsample_nearest -input $input -size {8}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-2.2 {Named parameter syntax with -scale_factor} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling with scale factor
    set result [torch::upsample_nearest -input $input -scale_factor {2.0}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-2.3 {Named parameter syntax with 4D tensor and 2D scale factor} {
    ;# Create 4D input tensor (N=1, C=1, H=2, W=2)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 2 2}]
    
    ;# Test upsampling with 2D scale factor
    set result [torch::upsample_nearest -input $input -scale_factor {2.0 2.0}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Test cases for camelCase alias
test upsample-nearest-3.1 {CamelCase alias with positional syntax} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling using camelCase alias
    set result [torch::upsampleNearest $input {8}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-3.2 {CamelCase alias with named parameters} {
    ;# Create 4D input tensor (N=1, C=1, H=2, W=2)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 2 2}]
    
    ;# Test upsampling using camelCase alias with named parameters
    set result [torch::upsampleNearest -input $input -size {4 4}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-3.3 {CamelCase alias with scale factor} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling using camelCase alias with scale factor
    set result [torch::upsampleNearest -input $input -scale_factor {2.0}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Error handling tests
test upsample-nearest-4.1 {Error handling - invalid input tensor} {
    ;# Test with non-existent tensor
    catch {torch::upsample_nearest invalid_tensor {4}} result
    expr {[string first "Invalid input tensor" $result] >= 0}
} {1}

test upsample-nearest-4.2 {Error handling - missing size parameter} {
    ;# Create 3D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test without size parameter
    catch {torch::upsample_nearest $input} result
    expr {[string first "Missing value for parameter" $result] >= 0}
} {1}

test upsample-nearest-4.3 {Error handling - invalid size list} {
    ;# Create 3D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test with invalid size list
    catch {torch::upsample_nearest $input "invalid_size"} result
    expr {[string first "Invalid size list element" $result] >= 0}
} {1}

test upsample-nearest-4.4 {Error handling - unknown named parameter} {
    ;# Create 3D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test with unknown parameter
    catch {torch::upsample_nearest -input $input -unknown_param value} result
    expr {[string first "Unknown parameter" $result] >= 0}
} {1}

test upsample-nearest-4.5 {Error handling - missing value for parameter} {
    ;# Create 3D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test with missing value
    catch {torch::upsample_nearest -input $input -size} result
    expr {[string first "Missing value for parameter" $result] >= 0}
} {1}

;# Mathematical correctness tests
test upsample-nearest-5.1 {Mathematical correctness - 3D tensor 1D upsampling} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Upsample from L=4 to L=8
    set result [torch::upsample_nearest $input {8}]
    
    ;# Get tensor shape to verify dimensions
    set shape [torch::tensor_shape $result]
    expr {$shape == "1 1 8"}
} {1}

test upsample-nearest-5.2 {Mathematical correctness - 4D tensor 2D upsampling} {
    ;# Create 4D input tensor (N=1, C=1, H=2, W=3)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 2 3}]
    
    ;# Upsample to H=4, W=6
    set result [torch::upsample_nearest $input {4 6}]
    
    ;# Get tensor shape to verify dimensions
    set shape [torch::tensor_shape $result]
    expr {$shape == "1 1 4 6"}
} {1}

test upsample-nearest-5.3 {Mathematical correctness - scale factor} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Upsample with scale factor of 2.0
    set result [torch::upsample_nearest -input $input -scale_factor {2.0}]
    
    ;# Get tensor shape to verify dimensions (4 * 2 = 8)
    set shape [torch::tensor_shape $result]
    expr {$shape == "1 1 8"}
} {1}

;# Data type support tests
test upsample-nearest-6.1 {Data type support - float32} {
    ;# Create 3D input tensor with float32
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling
    set result [torch::upsample_nearest $input {8}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-6.2 {Data type support - float64} {
    ;# Create 3D input tensor with float64
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float64 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling
    set result [torch::upsample_nearest $input {8}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-6.3 {Data type support - integer tensors not supported} {
    ;# Create 3D input tensor with int32 (nearest interpolation doesn't support integer tensors)
    set input_1d [torch::tensor_create {1 2 3 4} int32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling - should fail with integer tensors
    catch {torch::upsample_nearest $input {8}} result
    expr {[string first "not implemented" $result] >= 0 || [string first "Int" $result] >= 0}
} {1}

;# Edge case tests
test upsample-nearest-7.1 {Edge case - single element tensor} {
    ;# Create 3D tensor with single element (N=1, C=1, L=1)
    set input_1d [torch::tensor_create {1.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 1}]
    
    ;# Test upsampling
    set result [torch::upsample_nearest $input {5}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-7.2 {Edge case - large upsampling factor} {
    ;# Create 3D input tensor (N=1, C=1, L=2)
    set input_1d [torch::tensor_create {1.0 2.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 2}]
    
    ;# Test with large upsampling factor
    set result [torch::upsample_nearest $input {20}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test upsample-nearest-7.3 {Edge case - zero values in tensor} {
    ;# Create 3D input tensor with zeros (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {0.0 0.0 0.0 0.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test upsampling
    set result [torch::upsample_nearest $input {8}]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Syntax consistency tests
test upsample-nearest-8.1 {Syntax consistency - positional vs named} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test both syntaxes
    set result1 [torch::upsample_nearest $input {8}]
    set result2 [torch::upsample_nearest -input $input -size {8}]
    
    ;# Both should return valid tensor handles
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test upsample-nearest-8.2 {Syntax consistency - snake_case vs camelCase} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test both command names
    set result1 [torch::upsample_nearest $input {8}]
    set result2 [torch::upsampleNearest $input {8}]
    
    ;# Both should return valid tensor handles
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test upsample-nearest-8.3 {Syntax consistency - size vs scale_factor} {
    ;# Create 3D input tensor (N=1, C=1, L=4)
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {1 1 4}]
    
    ;# Test both size and scale_factor approaches
    set result1 [torch::upsample_nearest -input $input -size {8}]
    set result2 [torch::upsample_nearest -input $input -scale_factor {2.0}]
    
    ;# Both should return valid tensor handles
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

cleanupTests 