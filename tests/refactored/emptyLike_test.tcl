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

;# Test cases for positional syntax (backward compatibility)
test emptyLike-1.1 {Basic positional syntax} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test emptyLike-1.2 {Positional syntax with dtype} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like $input int64]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    return [list $shape $dtype]
} {{2 2} Int64}

test emptyLike-1.3 {Positional syntax with device} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like $input float32 cpu]
    set shape [torch::tensor_shape $result]
    set device [torch::tensor_device $result]
    return [list $shape $device]
} {{2 2} {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))}}

;# Test cases for named parameter syntax
test emptyLike-2.1 {Named parameter syntax with -input} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like -input $input]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test emptyLike-2.2 {Named parameter syntax with all parameters} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like -input $input -dtype float64 -device cpu -requiresGrad true]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $dtype $device $requires_grad]
} {{2 2} Float64 {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))} 1}

test emptyLike-2.3 {Named parameter syntax with different order} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like -dtype int32 -input $input -requiresGrad false -device cpu]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $dtype $device $requires_grad]
} {{2 2} Int32 {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))} 0}

;# Test cases for camelCase alias
test emptyLike-3.1 {CamelCase alias torch::emptyLike} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::emptyLike $input]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 2}

test emptyLike-3.2 {CamelCase alias with named parameters} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::emptyLike -input $input -dtype float32 -device cpu -requiresGrad false]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $dtype $device $requires_grad]
} {{2 2} Float32 {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))} 0}

;# Test error handling
test emptyLike-4.1 {Error: Missing arguments} {
    catch {torch::empty_like} result
    expr {[string first "Input tensor is required" $result] >= 0}
} {1}

test emptyLike-4.2 {Error: Invalid tensor name} {
    catch {torch::empty_like invalid_tensor} result
    expr {[string first "Invalid tensor name" $result] >= 0}
} {1}

test emptyLike-4.3 {Error: Too many positional arguments} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::empty_like $input float32 cpu extra} result
    expr {[string first "Invalid number of arguments" $result] >= 0}
} {1}

test emptyLike-4.4 {Error: Named parameter without value} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::empty_like -input} result
    expr {[string first "Missing value" $result] >= 0}
} {1}

test emptyLike-4.5 {Error: Unknown named parameter} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::empty_like -unknown $input} result
    expr {[string first "Unknown parameter" $result] >= 0}
} {1}

test emptyLike-4.6 {Error: Invalid requiresGrad value} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::empty_like -input $input -requiresGrad invalid} result
    expr {[string first "Invalid requiresGrad value" $result] >= 0}
} {1}

;# Test different input shapes and data types
test emptyLike-5.1 {1D tensor input} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    return $shape
} {5}

test emptyLike-5.2 {2D tensor input} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test emptyLike-5.3 {Different dtypes} {
    set input1 [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set input2 [torch::tensorCreate -data {{1 2} {3 4}} -dtype int64 -device cpu -requiresGrad false]
    set result1 [torch::empty_like -input $input1 -dtype float64]
    set result2 [torch::empty_like -input $input2 -dtype int32]
    set dtypes [list [torch::tensor_dtype $result1] [torch::tensor_dtype $result2]]
    return $dtypes
} {Float64 Int32}

;# Test edge cases
test emptyLike-6.1 {Edge case: single element tensor} {
    set input [torch::tensorCreate -data {5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::empty_like $input]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

;# Test syntax consistency
test emptyLike-7.1 {Syntax consistency: positional vs named} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::empty_like $input float64]
    set result2 [torch::empty_like -input $input -dtype float64]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    return [expr {$shape1 == $shape2 && $dtype1 == $dtype2}]
} {1}

test emptyLike-7.2 {Syntax consistency: snake_case vs camelCase} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::empty_like $input]
    set result2 [torch::emptyLike $input]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    return [expr {$shape1 == $shape2 && $dtype1 == $dtype2}]
} {1}

test emptyLike-7.3 {Syntax consistency: with parameters} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::empty_like $input int64]
    set result2 [torch::emptyLike -input $input -dtype int64]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    return [expr {$shape1 == $shape2 && $dtype1 == $dtype2}]
} {1}

cleanupTests 