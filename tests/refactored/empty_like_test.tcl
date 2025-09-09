#!/usr/bin/env tclsh

# Test file for torch::empty_like command with dual syntax support
# Tests both positional and named parameter syntax

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# =====================================================================
# TORCH::EMPTY_LIKE COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test empty_like-1.1 {Basic positional syntax} {
    set input [torch::ones {3 3}]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test empty_like-1.2 {Positional syntax with dtype} {
    set input [torch::ones {2 2}]
    set result [torch::empty_like $input float64]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test empty_like-1.3 {Positional syntax with dtype and device} {
    set input [torch::ones {2 3}]
    set result [torch::empty_like $input float32 cpu]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Tests for named parameter syntax
test empty_like-2.1 {Named parameter syntax basic} {
    set input [torch::ones {4 4}]
    set result [torch::empty_like -input $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4 4}

test empty_like-2.2 {Named parameter syntax with dtype} {
    set input [torch::ones {3 2}]
    set result [torch::empty_like -input $input -dtype float64]
    set shape [torch::tensor_shape $result]
    set shape
} {3 2}

test empty_like-2.3 {Named parameter syntax with device} {
    set input [torch::ones {2 5}]
    set result [torch::empty_like -input $input -device cpu]
    set shape [torch::tensor_shape $result]
    set shape
} {2 5}

test empty_like-2.4 {Named parameter syntax with requiresGrad} {
    set input [torch::ones {2 2}]
    set result [torch::empty_like -input $input -requiresGrad true]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test empty_like-2.5 {Named parameter syntax all parameters} {
    set input [torch::ones {3 3}]
    set result [torch::empty_like -input $input -dtype float32 -device cpu -requiresGrad false]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for camelCase alias
test empty_like-3.1 {CamelCase alias basic functionality} {
    set input [torch::ones {2 3}]
    set result [torch::emptyLike $input]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test empty_like-3.2 {CamelCase alias with named parameters} {
    set input [torch::ones {4 2}]
    set result [torch::emptyLike -input $input -dtype float64]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2}

test empty_like-3.3 {CamelCase alias with positional parameters} {
    set input [torch::ones {3 4}]
    set result [torch::emptyLike $input float32]
    set shape [torch::tensor_shape $result]
    set shape
} {3 4}

# Tests for error handling
test empty_like-4.1 {Error on missing input parameter} {
    catch {torch::empty_like} msg
    string match "*tensor*" $msg
} {1}

test empty_like-4.2 {Error on invalid tensor name} {
    catch {torch::empty_like invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test empty_like-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::empty_like -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test empty_like-4.4 {Error on unknown parameter} {
    set input [torch::ones {2 2}]
    catch {torch::empty_like -input $input -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test empty_like-4.5 {Error on missing parameter value} {
    set input [torch::ones {2 2}]
    catch {torch::empty_like -input $input -dtype} msg
    string match "*Missing value for parameter*" $msg
} {1}

test empty_like-4.6 {Error on invalid requiresGrad value} {
    set input [torch::ones {2 2}]
    catch {torch::empty_like -input $input -requiresGrad invalid} msg
    string match "*requiresGrad*" $msg
} {1}

# Tests for different tensor types and shapes
test empty_like-5.1 {1D tensor} {
    set input [torch::ones {5}]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test empty_like-5.2 {3D tensor} {
    set input [torch::ones {2 3 4}]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test empty_like-5.3 {Large tensor} {
    set input [torch::ones {10 10}]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test empty_like-5.4 {Different dtype input} {
    set input [torch::zeros {3 3}]
    set result [torch::empty_like -input $input -dtype float64]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for parameter flexibility
test empty_like-6.1 {Mixed parameter order} {
    set input [torch::ones {2 4}]
    set result [torch::empty_like -dtype float32 -input $input -device cpu]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

test empty_like-6.2 {Optional parameters} {
    set input [torch::ones {3 2}]
    set result1 [torch::empty_like -input $input]
    set result2 [torch::empty_like -input $input -dtype float32]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for syntax consistency
test empty_like-7.1 {Syntax consistency - shape preservation} {
    set input [torch::ones {4 3}]
    set result1 [torch::empty_like $input]
    set result2 [torch::empty_like -input $input]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test empty_like-7.2 {Syntax consistency - dtype parameter} {
    set input [torch::ones {2 5}]
    set result1 [torch::empty_like $input float64]
    set result2 [torch::empty_like -input $input -dtype float64]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for tensor properties
test empty_like-8.1 {Result is different from input} {
    set input [torch::ones {2 2}]
    set result [torch::empty_like $input]
    # Should be different tensor handles
    expr {$input ne $result}
} {1}

test empty_like-8.2 {Shape matches input} {
    set input [torch::ones {3 5 2}]
    set result [torch::empty_like $input]
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} {1}

test empty_like-8.3 {Different tensor values} {
    set input [torch::ones {2 2}]
    set result [torch::empty_like $input]
    # Should create valid tensor handle
    string match "tensor*" $result
} {1}

# Tests for requiresGrad parameter
test empty_like-9.1 {requiresGrad true} {
    set input [torch::ones {2 2}]
    set result [torch::empty_like -input $input -requiresGrad true]
    string match "tensor*" $result
} {1}

test empty_like-9.2 {requiresGrad false} {
    set input [torch::ones {2 2}]
    set result [torch::empty_like -input $input -requiresGrad false]
    string match "tensor*" $result
} {1}

# Cleanup tests
test empty_like-10.1 {Basic functionality verification} {
    set input [torch::ones {3 3}]
    set result [torch::empty_like $input]
    string match "tensor*" $result
} {1}

cleanupTests 