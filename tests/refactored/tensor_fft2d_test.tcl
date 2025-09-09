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

# Create test tensor (2D)
set tensor_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Test cases for positional syntax
test tensor-fft2d-1.1 {Basic positional syntax without dims} {
    set result [torch::tensor_fft2d $tensor_2d]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test tensor-fft2d-1.2 {Positional syntax with dims parameter} {
    set result [torch::tensor_fft2d $tensor_2d {0 1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

# Test cases for named parameter syntax
test tensor-fft2d-2.1 {Named parameter syntax with -tensor} {
    set result [torch::tensor_fft2d -tensor $tensor_2d]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test tensor-fft2d-2.2 {Named parameter syntax with -tensor and -dims} {
    set result [torch::tensor_fft2d -tensor $tensor_2d -dims {0 1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

# Test cases for camelCase alias
test tensor-fft2d-3.1 {CamelCase alias without dims} {
    set result [torch::tensorFft2d $tensor_2d]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test tensor-fft2d-3.2 {CamelCase alias with dims} {
    set result [torch::tensorFft2d $tensor_2d {0 1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test tensor-fft2d-3.3 {CamelCase alias with named parameters} {
    set result [torch::tensorFft2d -tensor $tensor_2d -dims {0 1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

# Error handling tests
test tensor-fft2d-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_fft2d invalid_tensor} result
    return [string match "*Invalid tensor name*" $result]
} {1}

test tensor-fft2d-4.2 {Error handling - dims not a list of 2} {
    catch {torch::tensor_fft2d $tensor_2d {0}} result
    return [string match "*dims must be a list of 2 integers*" $result]
} {1}

test tensor-fft2d-4.3 {Error handling - unknown named parameter} {
    catch {torch::tensor_fft2d -unknown param} result
    return [string match "*Unknown parameter*" $result]
} {1}

test tensor-fft2d-4.4 {Error handling - missing tensor parameter} {
    catch {torch::tensor_fft2d} result
    return [string match "*wrong # args*" $result]
} {1}

# Edge case: FFT on larger 2D tensor
test tensor-fft2d-5.1 {FFT on 3x3 tensor} {
    set t [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}} float32 cpu true]
    set result [torch::tensor_fft2d $t]
    set shape [torch::tensor_shape $result]
    return $shape
} {3 3}

cleanupTests 