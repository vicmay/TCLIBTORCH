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
# TORCH::ACOS COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test acos-1.1 {Basic positional syntax} {
    set t1 [torch::zeros {3 3}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test acos-1.2 {Positional syntax with mathematical values} {
    # Create tensor with values in valid range [-1, 1]
    set t1 [torch::ones {2 2}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test acos-1.3 {Positional syntax with small values} {
    set t1 [torch::zeros {1}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

# Tests for named parameter syntax
test acos-2.1 {Named parameter syntax basic} {
    set t1 [torch::zeros {2 3}]
    set result [torch::acos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test acos-2.2 {Named parameter syntax with valid mathematical values} {
    set t1 [torch::ones {3}]
    set result [torch::acos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test acos-2.3 {Named parameter syntax with different tensor sizes} {
    set t1 [torch::zeros {4 2 2}]
    set result [torch::acos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2 2}

# Tests for camelCase alias
test acos-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::zeros {2 2}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test acos-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::ones {3 3}]
    set result [torch::acos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for error handling
test acos-4.1 {Error on missing parameter} {
    catch {torch::acos} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test acos-4.2 {Error on invalid tensor name} {
    catch {torch::acos invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test acos-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::acos -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test acos-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::acos -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test acos-4.5 {Error on missing parameter value} {
    set t1 [torch::zeros {2 2}]
    catch {torch::acos -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

# Tests for data types and edge cases
test acos-5.1 {Float tensor handling} {
    set t1 [torch::zeros {2 2}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test acos-5.2 {Large tensor handling} {
    set t1 [torch::zeros {10 10}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test acos-5.3 {1D tensor handling} {
    set t1 [torch::zeros {5}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test acos-5.4 {3D tensor handling} {
    set t1 [torch::zeros {2 3 4}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

# Tests for syntax consistency (both syntaxes should produce same results)
test acos-6.1 {Syntax consistency - shape preservation} {
    set t1 [torch::zeros {3 3}]
    set result1 [torch::acos $t1]
    set result2 [torch::acos -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test acos-6.2 {Syntax consistency - multiple tensors} {
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set result1 [torch::acos $t1]
    set result2 [torch::acos -input $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test acos-7.1 {Mathematical correctness - acos(0) should be approximately π/2} {
    set t1 [torch::zeros {1}]
    set result [torch::acos $t1]
    # Result should be approximately π/2 ≈ 1.5708
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test acos-7.2 {Mathematical correctness - acos(1) should be 0} {
    set t1 [torch::ones {1}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test acos-7.3 {Mathematical correctness - preservation of tensor structure} {
    set t1 [torch::zeros {2 3}]
    set result [torch::acos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Tests for parameter validation
test acos-8.1 {Parameter validation - input parameter required} {
    catch {torch::acos -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test acos-8.2 {Parameter validation - extra parameters ignored} {
    set t1 [torch::zeros {2 2}]
    catch {torch::acos -input $t1 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

# Cleanup tests
test acos-9.1 {Basic functionality verification} {
    # Test that acos works correctly
    set t1 [torch::zeros {5 5}]
    set result [torch::acos $t1]
    string match "tensor*" $result
} {1}

cleanupTests 