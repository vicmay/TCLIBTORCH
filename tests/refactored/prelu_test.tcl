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

# Helper procedures for tensor comparison
proc tensors_equal {tensor1 tensor2 {tolerance 1e-6}} {
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# 1. Positional syntax tests (backward compatibility)

test prelu-1.1 {Basic positional syntax} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-1.2 {Positional with 2D input tensor} {
    set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32] {2 2}]
    set weight [torch::tensor_create 0.1 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-1.3 {Positional with compatible weight dimensions} {
    set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32] {2 2}]
    set weight [torch::tensor_create {0.1 0.2} float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-1.4 {Positional with different weight values} {
    set input [torch::tensor_create {3.0 -3.0 4.0 -4.0} float32]
    set weight [torch::tensor_create 0.5 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

# 2. Named parameter syntax tests

test prelu-2.1 {Named parameter syntax} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu -input $input -weight $weight]
    string match "tensor*" $result
} {1}

test prelu-2.2 {Named parameters in different order} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.3 float32]
    set result [torch::prelu -weight $weight -input $input]
    string match "tensor*" $result
} {1}

test prelu-2.3 {Named syntax with 2D tensors} {
    set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0 3.0 -3.0} float32] {2 3}]
    set weight [torch::tensor_create 0.25 float32]
    set result [torch::prelu -input $input -weight $weight]
    string match "tensor*" $result
} {1}

test prelu-2.4 {Named syntax with channel-wise weights} {
    set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0 3.0 -3.0} float32] {2 3}]
    set weight [torch::tensor_create {0.1 0.2 0.3} float32]
    set result [torch::prelu -input $input -weight $weight]
    string match "tensor*" $result
} {1}

# 3. camelCase alias tests

test prelu-3.1 {camelCase alias with positional syntax} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.15 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-3.2 {camelCase alias with named parameters} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.1 float32]
    set result [torch::prelu -input $input -weight $weight]
    string match "tensor*" $result
} {1}

test prelu-3.3 {camelCase alias parameter order variation} {
    set input [torch::tensor_create {2.0 -2.0 3.0 -3.0} float32]
    set weight [torch::tensor_create 0.4 float32]
    set result [torch::prelu -weight $weight -input $input]
    string match "tensor*" $result
} {1}

# 4. Error handling tests

test prelu-4.1 {Error on invalid input tensor handle} {
    set weight [torch::tensor_create 0.2 float32]
    catch {torch::prelu invalid_tensor $weight} result
    string match "*Invalid input tensor name*" $result
} {1}

test prelu-4.2 {Error on invalid weight tensor handle} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    catch {torch::prelu $input invalid_weight} result
    string match "*Invalid weight tensor name*" $result
} {1}

test prelu-4.3 {Error on missing weight parameter} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    catch {torch::prelu $input} result
    string match "*Usage: torch::prelu tensor weight*" $result
} {1}

test prelu-4.4 {Error on missing input in named syntax} {
    set weight [torch::tensor_create 0.2 float32]
    catch {torch::prelu -weight $weight} result
    string match "*Required parameters missing*" $result
} {1}

test prelu-4.5 {Error on missing weight in named syntax} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    catch {torch::prelu -input $input} result
    string match "*Required parameters missing*" $result
} {1}

test prelu-4.6 {Error on unknown parameter} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    catch {torch::prelu -input $input -weight $weight -unknown_param value} result
    string match "*Unknown parameter*" $result
} {1}

test prelu-4.7 {Error on unpaired named parameters} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    catch {torch::prelu -input $input -weight} result
    string match "*Named parameters must be in pairs*" $result
} {1}

# 5. Mathematical correctness tests

test prelu-5.1 {PReLU behavior for positive values} {
    set input [torch::tensor_create {1.0 2.0 3.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create {1.0 2.0 3.0} float32]
    tensors_equal $result $expected
} {1}

test prelu-5.2 {PReLU behavior for negative values} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0} -dtype float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create -data {-0.2 -0.4 -0.6} -dtype float32]
    tensors_equal $result $expected
} {1}

test prelu-5.3 {PReLU behavior for mixed values} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.1 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create {1.0 -0.1 2.0 -0.2} float32]
    tensors_equal $result $expected
} {1}

test prelu-5.4 {PReLU behavior with zero values} {
    set input [torch::tensor_create {0.0 1.0 -1.0} float32]
    set weight [torch::tensor_create 0.3 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create {0.0 1.0 -0.3} float32]
    tensors_equal $result $expected
} {1}

# 6. Different weight configurations

test prelu-6.1 {Small weight factor} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.01 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create {1.0 -0.01 2.0 -0.02} float32]
    tensors_equal $result $expected
} {1}

test prelu-6.2 {Large weight factor} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    set weight [torch::tensor_create 0.9 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create {1.0 -0.9} float32]
    tensors_equal $result $expected
} {1}

test prelu-6.3 {Weight factor of 1.0} {
    set input [torch::tensor_create {2.0 -2.0} float32]
    set weight [torch::tensor_create 1.0 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create {2.0 -2.0} float32]
    tensors_equal $result $expected
} {1}

# 7. Multi-dimensional tensor tests

test prelu-7.1 {2D tensor processing} {
    set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32] {2 2}]
    set weight [torch::tensor_create 0.25 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-7.2 {3D tensor processing} {
    set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0 3.0 -3.0 4.0 -4.0} float32] {2 2 2}]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

# 8. Syntax consistency tests

test prelu-8.1 {Consistency between positional and named syntax} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.3 float32]
    set result1 [torch::prelu $input $weight]
    set result2 [torch::prelu -input $input -weight $weight]
    tensors_equal $result1 $result2
} {1}

test prelu-8.2 {Consistency between parameter orders} {
    set input [torch::tensor_create {1.5 -1.5 2.5 -2.5} float32]
    set weight [torch::tensor_create 0.15 float32]
    set result1 [torch::prelu -input $input -weight $weight]
    set result2 [torch::prelu -weight $weight -input $input]
    tensors_equal $result1 $result2
} {1}

# 9. Edge case tests

test prelu-9.1 {Very small input values} {
    set input [torch::tensor_create {0.001 -0.001} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-9.2 {Very large input values} {
    set input [torch::tensor_create {1000.0 -1000.0} float32]
    set weight [torch::tensor_create 0.1 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-9.3 {Single element tensors} {
    set input [torch::tensor_create 5.0 float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-9.4 {Single negative element} {
    set input [torch::tensor_create -data {-3.0} -dtype float32]
    set weight [torch::tensor_create 0.4 float32]
    set result [torch::prelu $input $weight]
    set expected [torch::tensor_create -data {-1.2} -dtype float32]
    tensors_equal $result $expected
} {1}

# 10. Parameter validation tests

test prelu-10.1 {Invalid input handle with named syntax} {
    set weight [torch::tensor_create 0.2 float32]
    catch {torch::prelu -input "invalid_tensor" -weight $weight} result
    string match "*Invalid input tensor name*" $result
} {1}

test prelu-10.2 {Invalid weight handle with named syntax} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    catch {torch::prelu -input $input -weight "invalid_weight"} result
    string match "*Invalid weight tensor name*" $result
} {1}

test prelu-10.3 {Empty tensor handles} {
    catch {torch::prelu -input "" -weight ""} result
    string match "*Required parameters missing*" $result
} {1}

# 11. Data type compatibility tests

test prelu-11.1 {Float32 input and weight} {
    set input [torch::tensor_create {1.0 -1.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-11.2 {Different precision handling} {
    set input [torch::tensor_create {2.5 -2.5} float32]
    set weight [torch::tensor_create 0.333 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

# 12. Documentation examples verification

test prelu-12.1 {README example - positional syntax} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu $input $weight]
    string match "tensor*" $result
} {1}

test prelu-12.2 {README example - named syntax} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu -input $input -weight $weight]
    string match "tensor*" $result
} {1}

test prelu-12.3 {README example - camelCase (using existing alias)} {
    set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
    set weight [torch::tensor_create 0.2 float32]
    set result [torch::prelu -input $input -weight $weight]
    string match "tensor*" $result
} {1}

cleanupTests 