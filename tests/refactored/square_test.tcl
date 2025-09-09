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
test square-1.1 {Basic positional syntax} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 4.0 9.0}

test square-1.2 {Positional syntax with negative values} {
    set tensor [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {4.0 1.0 0.0 1.0 4.0}

test square-1.3 {Positional syntax with zero} {
    set tensor [torch::tensorCreate -data {0.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {0.0}

test square-1.4 {Positional syntax with large values} {
    set tensor [torch::tensorCreate -data {10.0 100.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {100.0 10000.0}

;# Test cases for named parameter syntax
test square-2.1 {Named parameter syntax with -input} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square -input $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 4.0 9.0}

test square-2.2 {Named parameter syntax with -tensor} {
    set tensor [torch::tensorCreate -data {2.0 3.0 4.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square -tensor $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {4.0 9.0 16.0}

test square-2.3 {Named parameter syntax with negative values} {
    set tensor [torch::tensorCreate -data {-3.0 -2.0 -1.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square -input $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {9.0 4.0 1.0}

test square-2.4 {Named parameter syntax with mixed values} {
    set tensor [torch::tensorCreate -data {-1.5 0.0 1.5} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square -tensor $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {2.25 0.0 2.25}

;# Test cases for camelCase alias
test square-3.1 {CamelCase alias torch::Square} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::Square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {1.0 4.0 9.0}

test square-3.2 {CamelCase alias with named parameters} {
    set tensor [torch::tensorCreate -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::Square -input $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {4.0 9.0}

;# Error handling tests
test square-4.1 {Error: Missing tensor argument} {
    catch {torch::square} result
    return $result
} {Usage: torch::square tensor | torch::square -input tensor}

test square-4.2 {Error: Invalid tensor name} {
    catch {torch::square invalid_tensor} result
    return $result
} {Invalid tensor name}

test square-4.3 {Error: Wrong number of positional arguments} {
    set tensor [torch::tensorCreate -data {1.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::square $tensor extra_arg} result
    return $result
} {Wrong number of positional arguments. Expected: torch::square tensor}

test square-4.4 {Error: Named parameter without value} {
    set tensor [torch::tensorCreate -data {1.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::square -input} result
    return $result
} {Named parameter requires a value}

test square-4.5 {Error: Unknown named parameter} {
    set tensor [torch::tensorCreate -data {1.0} -dtype float32 -device cpu -requiresGrad true]
    catch {torch::square -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

;# Test mathematical correctness
test square-5.1 {Mathematical correctness: squares of floats} {
    set tensor [torch::tensorCreate -data {0.5 1.5 2.5} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {0.25 2.25 6.25}

;# Test data type support
test square-6.1 {Data type support: int64} {
    set tensor [torch::tensorCreate -data {1 2 3} -dtype int64 -device cpu -requiresGrad false]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {1 4 9}

;# Test edge cases
test square-7.1 {Edge case: very large numbers} {
    set tensor [torch::tensorCreate -data {1e6 1e7} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {999999995904.0 100000000376832.0}

test square-7.2 {Edge case: very small numbers} {
    set tensor [torch::tensorCreate -data {1e-6 1e-7} -dtype float32 -device cpu -requiresGrad true]
    set result [torch::square $tensor]
    set values [torch::tensor_to_list $result]
    return $values
} {9.999999960041972e-13 9.9999998245167e-15}

;# Test syntax consistency
test square-8.1 {Syntax consistency: positional vs named} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::square $tensor]
    set result2 [torch::square -input $tensor]
    set values1 [torch::tensor_to_list $result1]
    set values2 [torch::tensor_to_list $result2]
    return [expr {$values1 == $values2}]
} {1}

test square-8.2 {Syntax consistency: snake_case vs camelCase} {
    set tensor [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set result1 [torch::square $tensor]
    set result2 [torch::Square $tensor]
    set values1 [torch::tensor_to_list $result1]
    set values2 [torch::tensor_to_list $result2]
    return [expr {$values1 == $values2}]
} {1}

cleanupTests 