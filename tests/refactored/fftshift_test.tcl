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
# Basic Positional Syntax Tests
# ============================================================================

test fftshift-1.1 {Basic positional syntax - 1D tensor, all dims} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-1.2 {Basic positional syntax - 1D tensor, specific dim} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftshift $input 0]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-1.3 {Basic positional syntax - 2D tensor, all dims} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 4}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

test fftshift-1.4 {Basic positional syntax - 2D tensor, specific dim} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 4}]
    set result [torch::fftshift $input 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

test fftshift-1.5 {Basic positional syntax - 3D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2}]
    set result [torch::fftshift $input 2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2 2}

# ============================================================================
# Named Parameter Syntax Tests
# ============================================================================

test fftshift-2.1 {Named parameter syntax - input only} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftshift -input $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-2.2 {Named parameter syntax - with dimension} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 4}]
    set result [torch::fftshift -input $input -dim 0]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

test fftshift-2.3 {Named parameter syntax - alternative parameter names} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftshift -tensor $input -dimension 0]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-2.4 {Named parameter syntax - parameter order independence} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftshift -dim 0 -input $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-2.5 {Named parameter syntax - 2D tensor with dim 1} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 4}]
    set result [torch::fftshift -input $input -dim 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

# ============================================================================
# CamelCase Alias Tests
# ============================================================================

test fftshift-3.1 {CamelCase alias - basic functionality} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftShift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-3.2 {CamelCase alias - with dimension} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftShift $input 0]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-3.3 {CamelCase alias - named parameters} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result [torch::fftShift -input $input -dim 0]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-3.4 {CamelCase alias - 2D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 4}]
    set result [torch::fftShift -input $input -dim 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4}

# ============================================================================
# Command Existence Tests
# ============================================================================

test fftshift-4.1 {Command torch::fftshift exists} {
    llength [info commands torch::fftshift]
} {1}

test fftshift-4.2 {Command torch::fftShift exists} {
    llength [info commands torch::fftShift]
} {1}

# ============================================================================
# Different Tensor Sizes Tests
# ============================================================================

test fftshift-5.1 {Small tensor - size 2} {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

test fftshift-5.2 {Odd-sized tensor - size 5} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -shape {5}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test fftshift-5.3 {Large tensor - size 8} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {8}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {8}

test fftshift-5.4 {3D tensor - different dimensions} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2}]
    set result1 [torch::fftshift $input 0]
    set result2 [torch::fftshift $input 1]
    set result3 [torch::fftshift $input 2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    list $shape1 $shape2 $shape3
} {{2 2 2} {2 2 2} {2 2 2}}

# ============================================================================
# Edge Cases Tests
# ============================================================================

test fftshift-6.1 {Single element tensor} {
    set input [torch::tensor_create -data {42.0} -shape {1}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test fftshift-6.2 {Negative values} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -4.0} -shape {4}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-6.3 {Zero values} {
    set input [torch::tensor_create -data {0.0 1.0 2.0 0.0} -shape {4}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-6.4 {Mixed positive and negative} {
    set input [torch::tensor_create -data {1.0 -2.0 3.0 -4.0} -shape {4}]
    set result [torch::fftshift $input]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

# ============================================================================
# Error Handling Tests
# ============================================================================

test fftshift-7.1 {Error - no arguments} {
    catch {torch::fftshift} result
    string match "*Usage*" $result
} {1}

test fftshift-7.2 {Error - invalid tensor} {
    catch {torch::fftshift invalid_tensor} result
    string match "*Error*" $result
} {1}

test fftshift-7.3 {Error - missing parameter value} {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2}]
    catch {torch::fftshift -input} result
    string match "*Missing value*" $result
} {1}

test fftshift-7.4 {Error - unknown parameter} {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2}]
    catch {torch::fftshift -unknown $input} result
    string match "*Unknown parameter*" $result
} {1}

test fftshift-7.5 {Error - invalid dimension value} {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2}]
    catch {torch::fftshift $input invalid_dim} result
    string match "*Error*" $result
} {1}

test fftshift-7.6 {Error - dimension out of range} {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2}]
    catch {torch::fftshift $input 5} result
    string match "*Error*" $result
} {1}

test fftshift-7.7 {Error - negative dimension beyond range} {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2}]
    catch {torch::fftshift $input -5} result
    string match "*Error*" $result
} {1}

test fftshift-7.8 {Error - too many positional arguments} {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2}]
    catch {torch::fftshift $input 0 extra} result
    string match "*Too many*" $result
} {1}

# ============================================================================
# Mathematical Correctness Tests
# ============================================================================

test fftshift-8.1 {Mathematical correctness - dimension preservation} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3}]
    set result [torch::fftshift $input]
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} {1}

test fftshift-8.2 {Mathematical correctness - dtype preservation} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]
    set result [torch::fftshift $input]
    set input_dtype [torch::tensor_dtype $input]
    set result_dtype [torch::tensor_dtype $result]
    expr {$input_dtype eq $result_dtype}
} {1}

test fftshift-8.3 {Mathematical correctness - different dtypes} {
    set input_int [torch::tensor_create -data {1 2 3 4} -shape {4} -dtype int32]
    set input_float [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float64]
    set result_int [torch::fftshift $input_int]
    set result_float [torch::fftshift $input_float]
    set dtype_int [torch::tensor_dtype $result_int]
    set dtype_float [torch::tensor_dtype $result_float]
    list $dtype_int $dtype_float
} {Int32 Float64}

# ============================================================================
# Syntax Equivalence Tests
# ============================================================================

test fftshift-9.1 {Syntax equivalence - positional vs named shapes} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result1 [torch::fftshift $input]
    set result2 [torch::fftshift -input $input]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test fftshift-9.2 {Syntax equivalence - with dimension shapes} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result1 [torch::fftshift $input 0]
    set result2 [torch::fftshift -input $input -dim 0]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test fftshift-9.3 {Syntax equivalence - camelCase vs snake_case shapes} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result1 [torch::fftshift $input]
    set result2 [torch::fftShift $input]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test fftshift-9.4 {Syntax equivalence - all forms shapes} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set result1 [torch::fftshift $input 0]
    set result2 [torch::fftshift -input $input -dim 0]
    set result3 [torch::fftShift $input 0]
    set result4 [torch::fftShift -input $input -dim 0]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    set shape4 [torch::tensor_shape $result4]
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4}
} {1}

# ============================================================================
# Performance and Integration Tests
# ============================================================================

test fftshift-10.1 {Integration with tensor creation} {
    set result [torch::fftshift [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test fftshift-10.2 {Chain operations} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4}]
    set shifted [torch::fftshift $input]
    set shifted_back [torch::fftshift $shifted]
    set original_shape [torch::tensor_shape $input]
    set final_shape [torch::tensor_shape $shifted_back]
    expr {$original_shape eq $final_shape}
} {1}

test fftshift-10.3 {Multiple dimensions sequentially} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 4}]
    set result1 [torch::fftshift $input 0]
    set result2 [torch::fftshift $result1 1]
    set shape [torch::tensor_shape $result2]
    set shape
} {2 4}

cleanupTests 