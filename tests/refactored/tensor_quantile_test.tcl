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

# Test cases for positional syntax
test tensor-quantile-1.1 {Basic positional syntax without dimension} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile $tensor 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 3.0) < 0.001}]
} {1}

test tensor-quantile-1.2 {Positional syntax with dimension} {
    set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result [torch::tensor_quantile $tensor 0.5 0]
    set value [torch::tensor_to_list $result]
    return [expr {abs([lindex $value 0] - 2.5) < 0.001 && abs([lindex $value 1] - 3.5) < 0.001 && abs([lindex $value 2] - 4.5) < 0.001}]
} {1}

test tensor-quantile-1.3 {Positional syntax with 0.25 quantile} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile $tensor 0.25]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 2.0) < 0.001}]
} {1}

test tensor-quantile-1.4 {Positional syntax with 0.75 quantile} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile $tensor 0.75]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 4.0) < 0.001}]
} {1}

# Test cases for named parameter syntax
test tensor-quantile-2.1 {Named parameter syntax without dimension} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile -input $tensor -q 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 3.0) < 0.001}]
} {1}

test tensor-quantile-2.2 {Named parameter syntax with dimension} {
    set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result [torch::tensor_quantile -input $tensor -q 0.5 -dim 0]
    set value [torch::tensor_to_list $result]
    return [expr {abs([lindex $value 0] - 2.5) < 0.001 && abs([lindex $value 1] - 3.5) < 0.001 && abs([lindex $value 2] - 4.5) < 0.001}]
} {1}

test tensor-quantile-2.3 {Named parameter syntax with -tensor alias} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile -tensor $tensor -q 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 3.0) < 0.001}]
} {1}

test tensor-quantile-2.4 {Named parameter syntax with -quantile alias} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile -input $tensor -quantile 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 3.0) < 0.001}]
} {1}

test tensor-quantile-2.5 {Named parameter syntax with -dimension alias} {
    set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result [torch::tensor_quantile -input $tensor -q 0.5 -dimension 0]
    set value [torch::tensor_to_list $result]
    return [expr {abs([lindex $value 0] - 2.5) < 0.001 && abs([lindex $value 1] - 3.5) < 0.001 && abs([lindex $value 2] - 4.5) < 0.001}]
} {1}

# Test cases for camelCase alias
test tensor-quantile-3.1 {CamelCase alias without dimension} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensorQuantile $tensor 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 3.0) < 0.001}]
} {1}

test tensor-quantile-3.2 {CamelCase alias with dimension} {
    set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result [torch::tensorQuantile $tensor 0.5 0]
    set value [torch::tensor_to_list $result]
    return [expr {abs([lindex $value 0] - 2.5) < 0.001 && abs([lindex $value 1] - 3.5) < 0.001 && abs([lindex $value 2] - 4.5) < 0.001}]
} {1}

test tensor-quantile-3.3 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensorQuantile -input $tensor -q 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 3.0) < 0.001}]
} {1}

# Error handling tests
test tensor-quantile-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_quantile invalid_tensor 0.5} result
    return [string match "*Invalid tensor name*" $result]
} {1}

test tensor-quantile-4.2 {Error handling - invalid quantile value (negative)} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_quantile $tensor -0.1} result
    return [string match "*Required parameters missing or invalid*" $result]
} {1}

test tensor-quantile-4.3 {Error handling - invalid quantile value (too large)} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_quantile $tensor 1.5} result
    return [string match "*Required parameters missing or invalid*" $result]
} {1}

test tensor-quantile-4.4 {Error handling - invalid dimension} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_quantile $tensor 0.5 10} result
    return [string match "*Dimension out of range*" $result]
} {1}

test tensor-quantile-4.5 {Error handling - missing required parameters} {
    catch {torch::tensor_quantile} result
    return [string match "*Required parameters missing or invalid*" $result]
} {1}

test tensor-quantile-4.6 {Error handling - unknown named parameter} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_quantile -input $tensor -q 0.5 -unknown param} result
    return [string match "*Unknown parameter*" $result]
} {1}

# Edge cases and mathematical correctness
test tensor-quantile-5.1 {Edge case - 0.0 quantile (minimum)} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile $tensor 0.0]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 1.0) < 0.001}]
} {1}

test tensor-quantile-5.2 {Edge case - 1.0 quantile (maximum)} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result [torch::tensor_quantile $tensor 1.0]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 5.0) < 0.001}]
} {1}

test tensor-quantile-5.3 {Mathematical correctness - 2D tensor with dimension 1} {
    set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result [torch::tensor_quantile $tensor 0.5 1]
    set value [torch::tensor_to_list $result]
    return [expr {abs([lindex $value 0] - 2.0) < 0.001 && abs([lindex $value 1] - 5.0) < 0.001}]
} {1}

test tensor-quantile-5.4 {Mathematical correctness - single element tensor} {
    set tensor [torch::tensor_create {42.0} float32 cpu true]
    set result [torch::tensor_quantile $tensor 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 42.0) < 0.001}]
} {1}

test tensor-quantile-5.5 {Mathematical correctness - negative values} {
    set tensor [torch::tensor_create -data {-5.0 -3.0 -1.0 1.0 3.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::tensor_quantile $tensor 0.5]
    set value [torch::tensor_to_list $result]
    return [expr {abs($value - 0.0) < 0.001}]
} {1}

# Consistency tests - both syntaxes should produce same results
test tensor-quantile-6.1 {Consistency - positional vs named syntax} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result1 [torch::tensor_quantile $tensor 0.5]
    set result2 [torch::tensor_quantile -input $tensor -q 0.5]
    set value1 [torch::tensor_to_list $result1]
    set value2 [torch::tensor_to_list $result2]
    return [expr {abs($value1 - $value2) < 0.001}]
} {1}

test tensor-quantile-6.2 {Consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]
    set result1 [torch::tensor_quantile $tensor 0.5]
    set result2 [torch::tensorQuantile $tensor 0.5]
    set value1 [torch::tensor_to_list $result1]
    set value2 [torch::tensor_to_list $result2]
    return [expr {abs($value1 - $value2) < 0.001}]
} {1}

cleanupTests 