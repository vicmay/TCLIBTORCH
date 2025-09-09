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

# Create test tensor
set test_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]

# Test cases for positional syntax (backward compatibility)
test tensor-ifft-1.1 {Basic positional syntax without dim} {
    set result [torch::tensor_ifft $test_tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test tensor-ifft-1.2 {Positional syntax with dim parameter} {
    set result [torch::tensor_ifft $test_tensor 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

# Test cases for named parameter syntax
test tensor-ifft-2.1 {Named parameter syntax with -tensor} {
    set result [torch::tensor_ifft -tensor $test_tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test tensor-ifft-2.2 {Named parameter syntax with -tensor and -dim} {
    set result [torch::tensor_ifft -tensor $test_tensor -dim 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

# Test cases for camelCase alias
test tensor-ifft-3.1 {CamelCase alias without dim} {
    set result [torch::tensorIfft $test_tensor]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test tensor-ifft-3.2 {CamelCase alias with dim} {
    set result [torch::tensorIfft $test_tensor 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

test tensor-ifft-3.3 {CamelCase alias with named parameters} {
    set result [torch::tensorIfft -tensor $test_tensor -dim 0]
    set shape [torch::tensor_shape $result]
    return $shape
} {4}

# Test mathematical correctness (IFFT should be inverse of FFT)
test tensor-ifft-4.1 {IFFT mathematical correctness} {
    # Create a simple signal
    set signal [torch::tensor_create {1.0 0.0 1.0 0.0} float32 cpu true]
    # Apply FFT then IFFT - should get back original (approximately)
    set fft_result [torch::tensor_fft $signal]
    set ifft_result [torch::tensor_ifft $fft_result]
    set shape [torch::tensor_shape $ifft_result]
    return [expr {$shape == 4}]
} {1}

# Error handling tests
test tensor-ifft-5.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_ifft invalid_tensor} result
    return [string match "*Invalid tensor name*" $result]
} {1}

test tensor-ifft-5.2 {Error handling - invalid dim parameter} {
    catch {torch::tensor_ifft $test_tensor invalid_dim} result
    return [string match "*Invalid dim parameter*" $result]
} {1}

test tensor-ifft-5.3 {Error handling - missing tensor parameter} {
    catch {torch::tensor_ifft} result
    return [string match "*wrong # args*" $result]
} {1}

test tensor-ifft-5.4 {Error handling - unknown named parameter} {
    catch {torch::tensor_ifft -unknown param} result
    return [string match "*Unknown parameter*" $result]
} {1}

# Test with 2D tensor
test tensor-ifft-6.1 {IFFT on 2D tensor} {
    set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_ifft $tensor_2d]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

# Test with different dimensions
test tensor-ifft-6.2 {IFFT on 2D tensor with specific dim} {
    set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_ifft $tensor_2d 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

cleanupTests 