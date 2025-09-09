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

# =====================================================================
# TORCH::CEIL COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test ceil-1.1 {Basic positional syntax} {
    set t1 [torch::zeros {3 3}]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test ceil-1.2 {Positional syntax with floating point values} {
    # Create tensor with floating point values that need ceiling
    set t1 [torch::tensor_create {1.2 2.7 -0.5 -1.8} float32]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test ceil-1.3 {Positional syntax with single value} {
    set t1 [torch::tensor_create {3.14} float32]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

# Tests for named parameter syntax
test ceil-2.1 {Named parameter syntax basic} {
    set t1 [torch::zeros {2 3}]
    set result [torch::ceil -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test ceil-2.2 {Named parameter syntax with tensor alias} {
    set t1 [torch::ones {3}]
    set result [torch::ceil -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test ceil-2.3 {Named parameter syntax with different tensor sizes} {
    set t1 [torch::zeros {4 2 2}]
    set result [torch::ceil -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2 2}

# Tests for camelCase alias (torch::ceil - no change needed)
test ceil-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::zeros {2 2}]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test ceil-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::ones {3 3}]
    set result [torch::ceil -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for error handling
test ceil-4.1 {Error on missing parameter} {
    catch {torch::ceil} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test ceil-4.2 {Error on invalid tensor name} {
    catch {torch::ceil invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test ceil-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::ceil -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test ceil-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::ceil -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test ceil-4.5 {Error on missing parameter value} {
    set t1 [torch::zeros {2 2}]
    catch {torch::ceil -input} msg
    string match "*Named parameter requires a value*" $msg
} {1}

# Tests for data types and edge cases
test ceil-5.1 {Float tensor handling} {
    set t1 [torch::tensor_create {1.1 2.9 3.0} float32]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test ceil-5.2 {Large tensor handling} {
    set t1 [torch::zeros {10 10}]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test ceil-5.3 {1D tensor handling} {
    set t1 [torch::tensor_create -data {-2.5 -1.1 0.7 1.0 2.3} -dtype float32]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test ceil-5.4 {3D tensor handling} {
    set t1 [torch::zeros {2 3 4}]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

# Tests for syntax consistency (both syntaxes should produce same results)
test ceil-6.1 {Syntax consistency - shape preservation} {
    set t1 [torch::zeros {3 3}]
    set result1 [torch::ceil $t1]
    set result2 [torch::ceil -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test ceil-6.2 {Syntax consistency - multiple tensors} {
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set result1 [torch::ceil $t1]
    set result2 [torch::ceil -input $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test ceil-7.1 {Mathematical correctness - ceil(0) should be 0} {
    set t1 [torch::zeros {1}]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test ceil-7.2 {Mathematical correctness - ceil(positive integers) should be unchanged} {
    set t1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test ceil-7.3 {Mathematical correctness - preservation of tensor structure} {
    set t1 [torch::ones {2 2}]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test ceil-7.4 {Mathematical correctness - negative values} {
    set t1 [torch::tensor_create -data {-2.5 -1.1 -0.1} -dtype float32]
    set result [torch::ceil $t1]
    # ceil(-2.5) = -2, ceil(-1.1) = -1, ceil(-0.1) = 0
    set shape [torch::tensor_shape $result]
    set shape
} {3}

# Tests for parameter validation
test ceil-8.1 {Parameter validation - input parameter required} {
    catch {torch::ceil -input} msg
    string match "*Named parameter requires a value*" $msg
} {1}

test ceil-8.2 {Parameter validation - extra parameters rejected} {
    set t1 [torch::zeros {2 2}]
    catch {torch::ceil -input $t1 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test ceil-8.3 {Parameter validation - wrong number of positional args} {
    set t1 [torch::zeros {2 2}]
    set t2 [torch::zeros {2 2}]
    catch {torch::ceil $t1 $t2} msg
    string match "*Wrong number of positional arguments*" $msg
} {1}

# Cleanup tests
test ceil-9.1 {Basic functionality verification} {
    # Test that ceil works correctly
    set t1 [torch::tensor_create {2.7} float32]
    set result [torch::ceil $t1]
    string match "tensor*" $result
} {1}

test ceil-9.2 {Edge case - very small positive values} {
    set t1 [torch::tensor_create {0.001 0.999} float32]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

test ceil-9.3 {Edge case - very small negative values} {
    set t1 [torch::tensor_create -data {-0.001 -0.999} -dtype float32]
    set result [torch::ceil $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

cleanupTests 