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
test empty-1.1 {Basic positional syntax} {
    set result [torch::empty {2 3}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test empty-1.2 {Positional syntax with dtype} {
    set result [torch::empty {2 3} int64]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    return [list $shape $dtype]
} {{2 3} Int64}

test empty-1.3 {Positional syntax with device} {
    set result [torch::empty {2 3} float32 cpu]
    set shape [torch::tensor_shape $result]
    set device [torch::tensor_device $result]
    return [list $shape $device]
} {{2 3} {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))}}

test empty-1.4 {Positional syntax with requires_grad} {
    set result [torch::empty {2 3} float32 cpu true]
    set shape [torch::tensor_shape $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $requires_grad]
} {{2 3} 1}

;# Test cases for named parameter syntax
test empty-2.1 {Named parameter syntax with -shape} {
    set result [torch::empty -shape {2 3}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test empty-2.2 {Named parameter syntax with all parameters (float32)} {
    set result [torch::empty -shape {2 3} -dtype float32 -device cpu -requiresGrad true]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $dtype $device $requires_grad]
} {{2 3} Float32 {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))} 1}

test empty-2.3 {Named parameter syntax with different order} {
    set result [torch::empty -dtype float64 -shape {3 4} -requiresGrad false -device cpu]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $dtype $device $requires_grad]
} {{3 4} Float64 {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))} 0}

;# Test cases for mixed syntax (positional + named)
test empty-3.1 {Mixed syntax: positional shape + named parameters} {
    set result [torch::empty {2 3} -dtype float32 -requiresGrad true]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $dtype $requires_grad]
} {{2 3} Float32 1}

;# Test cases for camelCase alias
test empty-4.1 {CamelCase alias torch::Empty} {
    set result [torch::Empty {2 3}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3}

test empty-4.2 {CamelCase alias with named parameters} {
    set result [torch::Empty -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    set requires_grad [torch::tensor_requires_grad $result]
    return [list $shape $dtype $device $requires_grad]
} {{2 3} Float32 {TensorOptions(dtype=float (default), device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))} 0}

;# Test error handling
test empty-5.1 {Error: Missing arguments} {
    catch {torch::empty} result
    expr {[string first "Invalid arguments" $result] >= 0}
} {1}

test empty-5.2 {Error: Invalid shape} {
    catch {torch::empty invalid_shape} result
    expr {[string first "expected list" $result] >= 0}
} {1}

test empty-5.3 {Error: Invalid dtype} {
    catch {torch::empty {2 3} invalid_dtype} result
    expr {[string first "Unknown scalar type" $result] >= 0}
} {1}

test empty-5.4 {Error: Invalid device} {
    catch {torch::empty {2 3} float32 invalid_device} result
    expr {[string first "Invalid device string" $result] >= 0}
} {1}

test empty-5.5 {Error: Named parameter without value} {
    catch {torch::empty -shape} result
    expr {[string first "Invalid arguments" $result] >= 0}
} {1}

test empty-5.6 {Error: Unknown named parameter} {
    catch {torch::empty -unknown {2 3}} result
    expr {[string first "Invalid arguments" $result] >= 0}
} {1}

test empty-5.7 {Error: Missing required -shape parameter} {
    catch {torch::empty -dtype float32} result
    expr {[string first "Invalid arguments" $result] >= 0}
} {1}

;# Test different shapes and data types
test empty-6.1 {1D tensor} {
    set result [torch::empty {5}]
    set shape [torch::tensor_shape $result]
    return $shape
} {5}

test empty-6.2 {3D tensor} {
    set result [torch::empty {2 3 4}]
    set shape [torch::tensor_shape $result]
    return $shape
} {2 3 4}

test empty-6.3 {Different dtypes} {
    set result1 [torch::empty {2 2} float32]
    set result2 [torch::empty {2 2} float64]
    set result3 [torch::empty {2 2} int32]
    set result4 [torch::empty {2 2} int64]
    set result5 [torch::empty {2 2} bool]
    set dtypes [list [torch::tensor_dtype $result1] [torch::tensor_dtype $result2] [torch::tensor_dtype $result3] [torch::tensor_dtype $result4] [torch::tensor_dtype $result5]]
    return $dtypes
} {Float32 Float64 Int32 Int64 Bool}

;# Test edge cases
test empty-7.1 {Edge case: single element} {
    set result [torch::empty {1}]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

;# Test syntax consistency
test empty-8.1 {Syntax consistency: positional vs named} {
    set result1 [torch::empty {2 3} float32 cpu true]
    set result2 [torch::empty -shape {2 3} -dtype float32 -device cpu -requiresGrad true]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    set device1 [torch::tensor_device $result1]
    set device2 [torch::tensor_device $result2]
    set grad1 [torch::tensor_requires_grad $result1]
    set grad2 [torch::tensor_requires_grad $result2]
    return [expr {$shape1 == $shape2 && $dtype1 == $dtype2 && $device1 == $device2 && $grad1 == $grad2}]
} {1}

test empty-8.2 {Syntax consistency: snake_case vs camelCase} {
    set result1 [torch::empty {2 3}]
    set result2 [torch::Empty {2 3}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    return [expr {$shape1 == $shape2 && $dtype1 == $dtype2}]
} {1}

cleanupTests 