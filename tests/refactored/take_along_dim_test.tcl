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
test take_along_dim-1.1 {Basic positional syntax} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim $input $indices]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 2.0 1.0 2.0 1.0 2.0}

test take_along_dim-1.2 {Positional syntax with dimension} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim $input $indices 1]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 2.0 1.0 5.0 4.0 5.0}

;# Test cases for named parameter syntax
test take_along_dim-2.1 {Named parameter syntax with -input and -indices} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim -input $input -indices $indices]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 2.0 1.0 2.0 1.0 2.0}

test take_along_dim-2.2 {Named parameter syntax with -dim} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim -input $input -indices $indices -dim 1]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 2.0 1.0 5.0 4.0 5.0}

test take_along_dim-2.3 {Named parameter syntax with different order} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim -dim 1 -indices $indices -input $input]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 2.0 1.0 5.0 4.0 5.0}

;# Test cases for camelCase alias
test take_along_dim-3.1 {CamelCase alias torch::takeAlongDim} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::takeAlongDim $input $indices]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 2.0 1.0 2.0 1.0 2.0}

test take_along_dim-3.2 {CamelCase alias with named parameters} {
    set input [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1 0} {1 0 1}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::takeAlongDim -input $input -indices $indices -dim 1]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 2.0 1.0 5.0 4.0 5.0}

;# Test error handling
test take_along_dim-4.1 {Error: Missing arguments} {
    catch {torch::take_along_dim} result
    expr {[string first "Usage" $result] >= 0}
} {1}

test take_along_dim-4.2 {Error: Missing indices} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::take_along_dim $input} result
    expr {[string first "Usage" $result] >= 0}
} {1}

test take_along_dim-4.3 {Error: Invalid input tensor} {
    set indices [torch::tensorCreate -data {0 1 0} -dtype int64 -device cpu -requiresGrad false]
    catch {torch::take_along_dim invalid_tensor $indices} result
    expr {[string first "Invalid input tensor" $result] >= 0}
} {1}

test take_along_dim-4.4 {Error: Invalid indices tensor} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::take_along_dim $input invalid_indices} result
    expr {[string first "Invalid indices tensor" $result] >= 0}
} {1}

test take_along_dim-4.5 {Error: Named parameter without value} {
    catch {torch::take_along_dim -input} result
    expr {[string first "Usage" $result] >= 0}
} {1}

test take_along_dim-4.6 {Error: Unknown named parameter} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {0 1 0} -dtype int64 -device cpu -requiresGrad false]
    catch {torch::take_along_dim -unknown $input -indices $indices} result
    expr {[string first "Unknown parameter" $result] >= 0}
} {1}

test take_along_dim-4.7 {Error: Invalid dim value} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {0 1 0} -dtype int64 -device cpu -requiresGrad false]
    catch {torch::take_along_dim $input $indices invalid_dim} result
    expr {[string first "Invalid dim value" $result] >= 0}
} {1}

;# Test mathematical correctness
test take_along_dim-5.1 {Mathematical correctness: 1D tensor} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {0 4 2 1 3} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim $input $indices]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 5.0 3.0 2.0 4.0}

test take_along_dim-5.2 {Mathematical correctness: 2D tensor along dimension 0} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0} {5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1} {2 0} {1 2}} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim $input $indices 0]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 4.0 5.0 2.0 3.0 6.0}

;# Test edge cases
test take_along_dim-6.1 {Edge case: single element} {
    set input [torch::tensorCreate -data {5.0} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {0} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::take_along_dim $input $indices]
    set values [torch::tensor_to_list $result]
    return $values
} {5.0}

;# Test syntax consistency
test take_along_dim-7.1 {Syntax consistency: positional vs named} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1} {1 0}} -dtype int64 -device cpu -requiresGrad false]
    set result1 [torch::take_along_dim $input $indices 1]
    set result2 [torch::take_along_dim -input $input -indices $indices -dim 1]
    set values1 [torch::tensor_to_list $result1]
    set values2 [torch::tensor_to_list $result2]
    return [expr {$values1 == $values2}]
} {1}

test take_along_dim-7.2 {Syntax consistency: snake_case vs camelCase} {
    set input [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32 -device cpu -requiresGrad true]
    set indices [torch::tensorCreate -data {{0 1} {1 0}} -dtype int64 -device cpu -requiresGrad false]
    set result1 [torch::take_along_dim $input $indices]
    set result2 [torch::takeAlongDim $input $indices]
    set values1 [torch::tensor_to_list $result1]
    set values2 [torch::tensor_to_list $result2]
    return [expr {$values1 == $values2}]
} {1}

cleanupTests 