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

# Test cases for positional syntax (backward compatibility)
test tensor_sqrt-1.1 {Basic positional syntax} {
    set t1 [torch::full {1} 4.0]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 2.0) < 0.001}
} {1}

test tensor_sqrt-1.2 {Positional syntax with larger value} {
    set t1 [torch::full {1} 25.0]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 5.0) < 0.001}
} {1}

test tensor_sqrt-1.3 {Positional syntax with zero} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 0.001}
} {1}

test tensor_sqrt-1.4 {Positional syntax with one} {
    set t1 [torch::ones {1}]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.0) < 0.001}
} {1}

test tensor_sqrt-1.5 {Positional syntax with multi-element tensor} {
    set t1 [torch::full {2 2} 16.0]
    set result [torch::tensor_sqrt $t1]
    set shape [torch::tensor_shape $result]
    # Check that shape is preserved
    expr {$shape eq "2 2"}
} {1}

# Test cases for named parameter syntax
test tensor_sqrt-2.1 {Named parameter syntax} {
    set t1 [torch::full {1} 9.0]
    set result [torch::tensor_sqrt -input $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 3.0) < 0.001}
} {1}

test tensor_sqrt-2.2 {Named parameter syntax with fractional values} {
    set t1 [torch::full {1} 0.25]
    set result [torch::tensor_sqrt -input $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.5) < 0.001}
} {1}

test tensor_sqrt-2.3 {Named parameter syntax preserves tensor properties} {
    set t1 [torch::full {2 3} 1.0]
    set result [torch::tensor_sqrt -input $t1]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "2 3" && [string match "*Float*" $dtype]}
} {1}

# Test cases for camelCase alias
test tensor_sqrt-3.1 {CamelCase alias syntax} {
    set t1 [torch::full {1} 36.0]
    set result [torch::tensorSqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 6.0) < 0.001}
} {1}

test tensor_sqrt-3.2 {CamelCase with named parameters} {
    set t1 [torch::full {1} 64.0]
    set result [torch::tensorSqrt -input $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 8.0) < 0.001}
} {1}

# Error handling tests
test tensor_sqrt-4.1 {Error: invalid tensor name (positional)} {
    catch {torch::tensor_sqrt invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_sqrt-4.2 {Error: invalid tensor name (named)} {
    catch {torch::tensor_sqrt -input invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_sqrt-4.3 {Error: missing required parameter} {
    catch {torch::tensor_sqrt} result
    expr {[string match "*Usage*" $result] || [string match "*Required parameter*" $result]}
} {1}

test tensor_sqrt-4.4 {Error: unknown parameter} {
    set t1 [torch::ones {1}]
    catch {torch::tensor_sqrt -invalid_param $t1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test tensor_sqrt-4.5 {Error: missing value for named parameter} {
    catch {torch::tensor_sqrt -input} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

# Mathematical correctness tests
test tensor_sqrt-5.1 {Mathematical properties: sqrt(x^2) = x for positive x} {
    set t1 [torch::full {1} 3.0]
    set squared [torch::tensor_mul $t1 $t1]
    set result [torch::tensor_sqrt $squared]
    set value [torch::tensor_item $result]
    expr {abs($value - 3.0) < 0.001}
} {1}

test tensor_sqrt-5.2 {Mathematical properties: sqrt(1) = 1} {
    set t1 [torch::ones {1}]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.0) < 0.001}
} {1}

test tensor_sqrt-5.3 {Mathematical properties: sqrt(0) = 0} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 0.001}
} {1}

# Data type consistency tests
test tensor_sqrt-6.1 {Different data types: float64} {
    set t1 [torch::full {1} 4.0 float64]
    set result [torch::tensor_sqrt $t1]
    set dtype [torch::tensor_dtype $result]
    set value [torch::tensor_item $result]
    expr {[string match "*Float64*" $dtype] && abs($value - 2.0) < 0.001}
} {1}

# Integration tests with other commands
test tensor_sqrt-7.1 {Integration with tensor creation} {
    set t1 [torch::full {1} 100.0]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 10.0) < 0.001}
} {1}

test tensor_sqrt-7.2 {Chain operations: sqrt then square} {
    set t1 [torch::full {1} 7.0]
    set sqrt_result [torch::tensor_sqrt $t1]
    set final_result [torch::tensor_mul $sqrt_result $sqrt_result]
    set value [torch::tensor_item $final_result]
    expr {abs($value - 7.0) < 0.001}
} {1}

# Syntax consistency tests
test tensor_sqrt-8.1 {Both syntaxes produce same result} {
    set t1 [torch::full {1} 144.0]
    set result1 [torch::tensor_sqrt $t1]
    set result2 [torch::tensor_sqrt -input $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {abs($value1 - $value2) < 0.001}
} {1}

test tensor_sqrt-8.2 {CamelCase alias produces same result} {
    set t1 [torch::full {1} 81.0]
    set result1 [torch::tensor_sqrt $t1]
    set result2 [torch::tensorSqrt $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {abs($value1 - $value2) < 0.001 && abs($value1 - 9.0) < 0.001}
} {1}

# Special mathematical cases
test tensor_sqrt-9.1 {Large numbers} {
    set t1 [torch::full {1} 10000.0]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 100.0) < 0.001}
} {1}

test tensor_sqrt-9.2 {Small positive numbers} {
    set t1 [torch::full {1} 0.01]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.1) < 0.001}
} {1}

test tensor_sqrt-9.3 {Perfect squares} {
    set t1 [torch::full {1} 49.0]
    set result [torch::tensor_sqrt $t1]
    set value [torch::tensor_item $result]
    expr {abs($value - 7.0) < 0.001}
} {1}

cleanupTests 