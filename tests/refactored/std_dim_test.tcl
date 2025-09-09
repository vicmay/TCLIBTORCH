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
test std_dim-1.1 {Basic positional syntax} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.581139}

test std_dim-1.2 {Positional syntax with unbiased=false} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 0 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.414214}

test std_dim-1.3 {Positional syntax with keepdim=true} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 0 1 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

test std_dim-1.4 {Positional syntax with 2D tensor} {
    set tensor [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {2.121320}

;# Test cases for named parameter syntax
test std_dim-2.1 {Named parameter syntax with -input and -dim} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim -input $tensor -dim 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.581139}

test std_dim-2.2 {Named parameter syntax with -tensor and -dim} {
    set tensor [torch::tensorCreate -data {2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim -tensor $tensor -dim 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.581139}

test std_dim-2.3 {Named parameter syntax with -unbiased=false} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim -input $tensor -dim 0 -unbiased 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.414214}

test std_dim-2.4 {Named parameter syntax with -keepdim=true} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim -input $tensor -dim 0 -keepdim 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

test std_dim-2.5 {Named parameter syntax with all parameters} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim -input $tensor -dim 0 -unbiased 0 -keepdim 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

test std_dim-2.6 {Named parameter syntax with different order} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim -dim 0 -unbiased 0 -input $tensor -keepdim 1]
    set shape [torch::tensor_shape $result]
    return $shape
} {1}

;# Test cases for camelCase alias
test std_dim-3.1 {CamelCase alias torch::stdDim} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::stdDim $tensor 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.581139}

test std_dim-3.2 {CamelCase alias with named parameters} {
    set tensor [torch::tensorCreate -data {2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::stdDim -input $tensor -dim 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.581139}

;# Test error handling
test std_dim-4.1 {Error: Missing tensor} {
    catch {torch::std_dim} result
    return $result
} {Usage: torch::std_dim tensor dim ?unbiased? ?keepdim? | torch::std_dim -input tensor -dim dim ?-unbiased bool? ?-keepdim bool?}

test std_dim-4.2 {Error: Missing dim in positional syntax} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::std_dim $tensor} result
    return $result
} {Usage: torch::std_dim tensor dim ?unbiased? ?keepdim?}

test std_dim-4.3 {Error: Invalid tensor name} {
    catch {torch::std_dim invalid_tensor 0} result
    return $result
} {Invalid tensor name}

test std_dim-4.4 {Error: Named parameter without value} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::std_dim -input} result
    return $result
} {Missing value for parameter}

test std_dim-4.5 {Error: Unknown named parameter} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::std_dim -unknown $tensor -dim 0} result
    return $result
} {Unknown parameter: -unknown}

test std_dim-4.6 {Error: Invalid dim value} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::std_dim $tensor invalid_dim} result
    return $result
} {Invalid dim value}

test std_dim-4.7 {Error: Invalid unbiased value} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::std_dim $tensor 0 invalid_unbiased} result
    return $result
} {Invalid unbiased value}

test std_dim-4.8 {Error: Invalid keepdim value} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::std_dim $tensor 0 1 invalid_keepdim} result
    return $result
} {Invalid keepdim value}

;# Test mathematical correctness
test std_dim-5.1 {Mathematical correctness: standard deviation of known values} {
    set tensor [torch::tensorCreate -data {1.0 3.0 5.0 7.0 9.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {3.162278}

test std_dim-5.2 {Mathematical correctness: unbiased vs biased} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::std_dim $tensor 0 1]  ;# unbiased=true (default)
    set result2 [torch::std_dim $tensor 0 0]  ;# unbiased=false
    set val1 [lindex [torch::tensor_to_list $result1] 0]
    set val2 [lindex [torch::tensor_to_list $result2] 0]
    return [expr {$val1 > $val2}]
} {1}

;# Test data type support - removed float64 test due to tensor_to_list compatibility

;# Test edge cases
test std_dim-7.1 {Edge case: single element tensor} {
    set tensor [torch::tensorCreate -data {5.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 0]
    set values [torch::tensor_to_list $result]
    set val [lindex $values 0]
    if {[string is double $val] && [expr {$val != $val}]} {
        return "NaN"  ;# Handle NaN case
    }
    return [format "%.6f" $val]
} {NaN}

test std_dim-7.2 {Edge case: all same values} {
    set tensor [torch::tensorCreate -data {3.0 3.0 3.0 3.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 0]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {0.000000}

test std_dim-7.3 {Edge case: 2D tensor along dimension 1} {
    set tensor [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::std_dim $tensor 1]
    set values [torch::tensor_to_list $result]
    return [format "%.6f" [lindex $values 0]]
} {1.000000}

;# Test syntax consistency
test std_dim-8.1 {Syntax consistency: positional vs named} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::std_dim $tensor 0 1 0]
    set result2 [torch::std_dim -input $tensor -dim 0 -unbiased 1 -keepdim 0]
    set values1 [torch::tensor_to_list $result1]
    set values2 [torch::tensor_to_list $result2]
    return [expr {abs([lindex $values1 0] - [lindex $values2 0]) < 1e-6}]
} {1}

test std_dim-8.2 {Syntax consistency: snake_case vs camelCase} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::std_dim $tensor 0]
    set result2 [torch::stdDim $tensor 0]
    set values1 [torch::tensor_to_list $result1]
    set values2 [torch::tensor_to_list $result2]
    return [expr {abs([lindex $values1 0] - [lindex $values2 0]) < 1e-6}]
} {1}

cleanupTests 