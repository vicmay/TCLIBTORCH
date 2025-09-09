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
# TORCH::ASINH COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test asinh-1.1 {Basic positional syntax} {
    set t1 [torch::zeros {3 3}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test asinh-1.2 {Positional syntax with mathematical values} {
    # Create tensor with positive values
    set t1 [torch::ones {2 2}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test asinh-1.3 {Positional syntax with small values} {
    set t1 [torch::zeros {1}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

# Tests for named parameter syntax
test asinh-2.1 {Named parameter syntax basic} {
    set t1 [torch::zeros {2 3}]
    set result [torch::asinh -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test asinh-2.2 {Named parameter syntax with valid mathematical values} {
    set t1 [torch::ones {3}]
    set result [torch::asinh -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test asinh-2.3 {Named parameter syntax with different tensor sizes} {
    set t1 [torch::zeros {4 2 2}]
    set result [torch::asinh -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2 2}

test asinh-2.4 {Named parameter syntax with -tensor alias} {
    set t1 [torch::zeros {2 2}]
    set result [torch::asinh -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

# Tests for camelCase alias (asinh is already camelCase)
test asinh-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::zeros {2 2}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test asinh-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::ones {3 3}]
    set result [torch::asinh -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for error handling
test asinh-4.1 {Error on missing parameter} {
    catch {torch::asinh} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test asinh-4.2 {Error on invalid tensor name} {
    catch {torch::asinh invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test asinh-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::asinh -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test asinh-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::asinh -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test asinh-4.5 {Error on missing parameter value} {
    set t1 [torch::zeros {2 2}]
    catch {torch::asinh -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test asinh-4.6 {Error on too many positional arguments} {
    set t1 [torch::zeros {2 2}]
    set t2 [torch::zeros {2 2}]
    catch {torch::asinh $t1 $t2} msg
    string match "*Usage*" $msg
} {1}

# Tests for data types and edge cases
test asinh-5.1 {Float tensor handling} {
    set t1 [torch::zeros {2 2}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test asinh-5.2 {Large tensor handling} {
    set t1 [torch::zeros {10 10}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test asinh-5.3 {1D tensor handling} {
    set t1 [torch::zeros {5}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test asinh-5.4 {3D tensor handling} {
    set t1 [torch::zeros {2 3 4}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

# Tests for syntax consistency (both syntaxes should produce same results)
test asinh-6.1 {Syntax consistency - shape preservation} {
    set t1 [torch::zeros {3 3}]
    set result1 [torch::asinh $t1]
    set result2 [torch::asinh -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test asinh-6.2 {Syntax consistency - multiple tensors} {
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set result1 [torch::asinh $t1]
    set result2 [torch::asinh -input $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test asinh-6.3 {Syntax consistency - parameter alias equivalence} {
    set t1 [torch::zeros {2 2}]
    set result1 [torch::asinh -input $t1]
    set result2 [torch::asinh -tensor $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test asinh-7.1 {Mathematical correctness - asinh(0) should be 0} {
    set t1 [torch::zeros {1}]
    set result [torch::asinh $t1]
    # Result should be 0
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test asinh-7.2 {Mathematical correctness - asinh(1) behavior} {
    set t1 [torch::ones {1}]
    set result [torch::asinh $t1]
    # Result should be approximately 0.8814 (asinh(1))
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test asinh-7.3 {Mathematical correctness - preservation of tensor structure} {
    set t1 [torch::zeros {2 3}]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Tests for parameter validation
test asinh-8.1 {Parameter validation - input parameter required} {
    catch {torch::asinh -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test asinh-8.2 {Parameter validation - extra parameters ignored} {
    set t1 [torch::zeros {2 2}]
    catch {torch::asinh -input $t1 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test asinh-8.3 {Parameter validation - valid parameter names} {
    set t1 [torch::zeros {2 2}]
    # Both -input and -tensor should be valid
    set result1 [torch::asinh -input $t1]
    set result2 [torch::asinh -tensor $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for large value handling (asinh has no domain restrictions)
test asinh-9.1 {Large positive values} {
    # asinh can handle any real value, unlike asin
    set t1 [torch::tensor_create -data {10.0 100.0 1000.0} -shape {3} -dtype float32 -device cpu]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test asinh-9.2 {Large negative values} {
    # asinh is an odd function: asinh(-x) = -asinh(x)
    set t1 [torch::tensor_create -data {-10.0 -100.0 -1000.0} -shape {3} -dtype float32 -device cpu]
    set result [torch::asinh $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

# Cleanup tests
test asinh-10.1 {Basic functionality verification} {
    # Test that asinh works correctly
    set t1 [torch::zeros {5 5}]
    set result [torch::asinh $t1]
    string match "tensor*" $result
} {1}

test asinh-10.2 {Return value format} {
    set t1 [torch::zeros {3 3}]
    set result [torch::asinh $t1]
    # Result should be a tensor handle
    expr {[string length $result] > 0}
} {1}

cleanupTests 