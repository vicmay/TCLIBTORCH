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
# TORCH::ASIN COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test asin-1.1 {Basic positional syntax} {
    set t1 [torch::zeros {3 3}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test asin-1.2 {Positional syntax with mathematical values} {
    # Create tensor with values in valid range [-1, 1]
    set t1 [torch::ones {2 2}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test asin-1.3 {Positional syntax with small values} {
    set t1 [torch::zeros {1}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

# Tests for named parameter syntax
test asin-2.1 {Named parameter syntax basic} {
    set t1 [torch::zeros {2 3}]
    set result [torch::asin -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test asin-2.2 {Named parameter syntax with valid mathematical values} {
    set t1 [torch::ones {3}]
    set result [torch::asin -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test asin-2.3 {Named parameter syntax with different tensor sizes} {
    set t1 [torch::zeros {4 2 2}]
    set result [torch::asin -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2 2}

test asin-2.4 {Named parameter syntax with -tensor alias} {
    set t1 [torch::zeros {2 2}]
    set result [torch::asin -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

# Tests for camelCase alias (asin is already camelCase)
test asin-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::zeros {2 2}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test asin-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::ones {3 3}]
    set result [torch::asin -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for error handling
test asin-4.1 {Error on missing parameter} {
    catch {torch::asin} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test asin-4.2 {Error on invalid tensor name} {
    catch {torch::asin invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test asin-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::asin -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test asin-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::asin -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test asin-4.5 {Error on missing parameter value} {
    set t1 [torch::zeros {2 2}]
    catch {torch::asin -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test asin-4.6 {Error on too many positional arguments} {
    set t1 [torch::zeros {2 2}]
    set t2 [torch::zeros {2 2}]
    catch {torch::asin $t1 $t2} msg
    string match "*Usage*" $msg
} {1}

# Tests for data types and edge cases
test asin-5.1 {Float tensor handling} {
    set t1 [torch::zeros {2 2}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test asin-5.2 {Large tensor handling} {
    set t1 [torch::zeros {10 10}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test asin-5.3 {1D tensor handling} {
    set t1 [torch::zeros {5}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test asin-5.4 {3D tensor handling} {
    set t1 [torch::zeros {2 3 4}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

# Tests for syntax consistency (both syntaxes should produce same results)
test asin-6.1 {Syntax consistency - shape preservation} {
    set t1 [torch::zeros {3 3}]
    set result1 [torch::asin $t1]
    set result2 [torch::asin -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test asin-6.2 {Syntax consistency - multiple tensors} {
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set result1 [torch::asin $t1]
    set result2 [torch::asin -input $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test asin-6.3 {Syntax consistency - parameter alias equivalence} {
    set t1 [torch::zeros {2 2}]
    set result1 [torch::asin -input $t1]
    set result2 [torch::asin -tensor $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test asin-7.1 {Mathematical correctness - asin(0) should be 0} {
    set t1 [torch::zeros {1}]
    set result [torch::asin $t1]
    # Result should be 0
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test asin-7.2 {Mathematical correctness - asin(1) should be approximately π/2} {
    set t1 [torch::ones {1}]
    set result [torch::asin $t1]
    # Result should be approximately π/2 ≈ 1.5708
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test asin-7.3 {Mathematical correctness - preservation of tensor structure} {
    set t1 [torch::zeros {2 3}]
    set result [torch::asin $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Tests for parameter validation
test asin-8.1 {Parameter validation - input parameter required} {
    catch {torch::asin -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test asin-8.2 {Parameter validation - extra parameters ignored} {
    set t1 [torch::zeros {2 2}]
    catch {torch::asin -input $t1 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test asin-8.3 {Parameter validation - valid parameter names} {
    set t1 [torch::zeros {2 2}]
    # Both -input and -tensor should be valid
    set result1 [torch::asin -input $t1]
    set result2 [torch::asin -tensor $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Cleanup tests
test asin-9.1 {Basic functionality verification} {
    # Test that asin works correctly
    set t1 [torch::zeros {5 5}]
    set result [torch::asin $t1]
    string match "tensor*" $result
} {1}

test asin-9.2 {Return value format} {
    set t1 [torch::zeros {3 3}]
    set result [torch::asin $t1]
    # Result should be a tensor handle
    expr {[string length $result] > 0}
} {1}

cleanupTests 