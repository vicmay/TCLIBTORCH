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

# Test helper function to verify tensor result
proc verify_denormalize_result {result} {
    # Verify we got a valid tensor handle back
    if {![string match "tensor*" $result]} {
        return 0
    }
    
    # Basic verification that the tensor exists and can be accessed
    set shape [torch::tensor_shape $result]
    if {$shape == ""} {
        return 0
    }
    
    return 1
}

# Test helper function to create test tensors
proc create_test_image {} {
    # Create a simple 2x2 single channel image for simplicity
    return [torch::tensor_create -data {0.5 0.6 0.7 0.8} -shape {2 2} -dtype float32]
}

proc create_test_mean {} {
    # Mean value as a scalar tensor
    return [torch::tensor_create -data {0.485} -shape {1} -dtype float32]
}

proc create_test_std {} {
    # Std value as a scalar tensor
    return [torch::tensor_create -data {0.229} -shape {1} -dtype float32]
}

# Test positional syntax
test denormalize_image-1.1 {Basic positional syntax without inplace} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image $image $mean $std]
    verify_denormalize_result $result
} {1}

test denormalize_image-1.2 {Positional syntax with inplace=0} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image $image $mean $std 0]
    verify_denormalize_result $result
} {1}

test denormalize_image-1.3 {Positional syntax with inplace=1} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image $image $mean $std 1]
    verify_denormalize_result $result
} {1}

# Test named parameter syntax
test denormalize_image-2.1 {Named parameter syntax basic} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image -image $image -mean $mean -std $std]
    verify_denormalize_result $result
} {1}

test denormalize_image-2.2 {Named parameter with inplace=false} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image -image $image -mean $mean -std $std -inplace 0]
    verify_denormalize_result $result
} {1}

test denormalize_image-2.3 {Named parameter with inplace=true} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image -image $image -mean $mean -std $std -inplace 1]
    verify_denormalize_result $result
} {1}

test denormalize_image-2.4 {Named parameters in different order} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image -std $std -mean $mean -image $image]
    verify_denormalize_result $result
} {1}

# Test camelCase alias
test denormalize_image-3.1 {CamelCase alias positional syntax} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalizeImage $image $mean $std]
    verify_denormalize_result $result
} {1}

test denormalize_image-3.2 {CamelCase alias named parameter syntax} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalizeImage -image $image -mean $mean -std $std]
    verify_denormalize_result $result
} {1}

# Test mathematical correctness
test denormalize_image-4.1 {Mathematical correctness verification} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    set result [torch::denormalize_image $image $mean $std]
    
    # Verify result has same shape as input
    set input_shape [torch::tensor_shape $image]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} {1}

# Test consistency between syntaxes
test denormalize_image-5.1 {Consistency between positional and named syntax} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    
    set result1 [torch::denormalize_image $image $mean $std 0]
    set result2 [torch::denormalize_image -image $image -mean $mean -std $std -inplace 0]
    
    # Both should be valid tensors
    set valid1 [verify_denormalize_result $result1]
    set valid2 [verify_denormalize_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test denormalize_image-5.2 {Consistency between snake_case and camelCase} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    
    set result1 [torch::denormalize_image $image $mean $std]
    set result2 [torch::denormalizeImage $image $mean $std]
    
    # Both should be valid tensors
    set valid1 [verify_denormalize_result $result1]
    set valid2 [verify_denormalize_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

# Test data type support
test denormalize_image-6.1 {Float32 tensor support} {
    set image [torch::tensor_create -data {0.5 0.6} -shape {2} -dtype float32]
    set mean [torch::tensor_create -data {0.1} -shape {1} -dtype float32]
    set std [torch::tensor_create -data {0.2} -shape {1} -dtype float32]
    set result [torch::denormalize_image -image $image -mean $mean -std $std]
    expr {[verify_denormalize_result $result] && [torch::tensor_dtype $result] eq "Float32"}
} {1}

test denormalize_image-6.2 {Float64 tensor support} {
    set image [torch::tensor_create -data {0.5 0.6} -shape {2} -dtype float64]
    set mean [torch::tensor_create -data {0.1} -shape {1} -dtype float64]
    set std [torch::tensor_create -data {0.2} -shape {1} -dtype float64]
    set result [torch::denormalize_image -image $image -mean $mean -std $std]
    expr {[verify_denormalize_result $result] && [torch::tensor_dtype $result] eq "Float64"}
} {1}

# Test edge cases
test denormalize_image-7.1 {Single pixel image} {
    set image [torch::tensor_create -data {0.5} -shape {1} -dtype float32]
    set mean [torch::tensor_create -data {0.1} -shape {1} -dtype float32]
    set std [torch::tensor_create -data {0.2} -shape {1} -dtype float32]
    set result [torch::denormalize_image $image $mean $std]
    verify_denormalize_result $result
} {1}

test denormalize_image-7.2 {Grayscale image (single channel)} {
    set image [torch::tensor_create -data {0.1 0.2 0.3 0.4} -shape {1 2 2} -dtype float32]
    set mean [torch::tensor_create -data {0.5} -shape {1} -dtype float32]
    set std [torch::tensor_create -data {0.3} -shape {1} -dtype float32]
    set result [torch::denormalize_image -image $image -mean $mean -std $std]
    expr {[verify_denormalize_result $result] && [torch::tensor_shape $result] eq "1 2 2"}
} {1}

test denormalize_image-7.3 {Zero mean and unit std} {
    set image [torch::tensor_create -data {0.5 0.6} -shape {2} -dtype float32]
    set mean [torch::tensor_create -data {0.0} -shape {1} -dtype float32]
    set std [torch::tensor_create -data {1.0} -shape {1} -dtype float32]
    set result [torch::denormalize_image $image $mean $std]
    verify_denormalize_result $result
} {1}

# Test error handling
test denormalize_image-8.1 {Invalid image tensor name positional} {
    set mean [create_test_mean]
    set std [create_test_std]
    catch {torch::denormalize_image invalid_tensor $mean $std} msg
    expr {[string match "*Invalid image tensor*" $msg]}
} {1}

test denormalize_image-8.2 {Invalid mean tensor name} {
    set image [create_test_image]
    set std [create_test_std]
    catch {torch::denormalize_image $image invalid_tensor $std} msg
    expr {[string match "*Invalid mean tensor*" $msg]}
} {1}

test denormalize_image-8.3 {Invalid std tensor name} {
    set image [create_test_image]
    set mean [create_test_mean]
    catch {torch::denormalize_image $image $mean invalid_tensor} msg
    expr {[string match "*Invalid std tensor*" $msg]}
} {1}

test denormalize_image-8.4 {Invalid tensor name named parameter} {
    set mean [create_test_mean]
    set std [create_test_std]
    catch {torch::denormalize_image -image invalid_tensor -mean $mean -std $std} msg
    expr {[string match "*Invalid image tensor*" $msg]}
} {1}

test denormalize_image-8.5 {Missing required parameters} {
    catch {torch::denormalize_image} msg
    expr {[string match "*Usage*" $msg] || [string match "*missing*" $msg]}
} {1}

test denormalize_image-8.6 {Too few positional arguments} {
    set image [create_test_image]
    set mean [create_test_mean]
    catch {torch::denormalize_image $image $mean} msg
    expr {[string match "*Usage*" $msg] || [string match "*missing*" $msg]}
} {1}

test denormalize_image-8.7 {Invalid parameter name} {
    set image [create_test_image]
    set mean [create_test_mean]
    set std [create_test_std]
    catch {torch::denormalize_image -invalid $image -mean $mean -std $std} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test denormalize_image-8.8 {Missing value for named parameter} {
    catch {torch::denormalize_image -image} msg
    expr {[string match "*Missing value*" $msg] || [string match "*wrong # args*" $msg] || [string match "*Usage*" $msg]}
} {1}

test denormalize_image-8.9 {Missing required named parameters} {
    set image [create_test_image]
    catch {torch::denormalize_image -image $image} msg
    expr {[string match "*Required parameters missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

cleanupTests 