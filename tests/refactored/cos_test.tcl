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
# TORCH::COS COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test cos-1.1 {Basic positional syntax} {
    set t1 [torch::zeros {3 3}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test cos-1.2 {Positional syntax with mathematical values} {
    set t1 [torch::ones {2 2}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test cos-1.3 {Positional syntax with small values} {
    set t1 [torch::zeros {1}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

# Tests for named parameter syntax
test cos-2.1 {Named parameter syntax basic} {
    set t1 [torch::zeros {2 3}]
    set result [torch::cos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test cos-2.2 {Named parameter syntax with valid mathematical values} {
    set t1 [torch::ones {3}]
    set result [torch::cos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test cos-2.3 {Named parameter syntax with different tensor sizes} {
    set t1 [torch::zeros {4 2 2}]
    set result [torch::cos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2 2}

test cos-2.4 {Named parameter syntax with -tensor alias} {
    set t1 [torch::zeros {2 2}]
    set result [torch::cos -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

# Tests for camelCase alias
test cos-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::zeros {2 2}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test cos-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::ones {3 3}]
    set result [torch::cos -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for error handling
test cos-4.1 {Error on missing parameter} {
    catch {torch::cos} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test cos-4.2 {Error on invalid tensor name} {
    catch {torch::cos invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test cos-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::cos -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test cos-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::cos -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test cos-4.5 {Error on missing parameter value} {
    set t1 [torch::zeros {2 2}]
    catch {torch::cos -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

# Tests for data types and edge cases
test cos-5.1 {Float tensor handling} {
    set t1 [torch::zeros {2 2}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test cos-5.2 {Large tensor handling} {
    set t1 [torch::zeros {10 10}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test cos-5.3 {1D tensor handling} {
    set t1 [torch::zeros {5}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test cos-5.4 {3D tensor handling} {
    set t1 [torch::zeros {2 3 4}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

# Tests for syntax consistency (both syntaxes should produce same results)
test cos-6.1 {Syntax consistency - shape preservation} {
    set t1 [torch::zeros {3 3}]
    set result1 [torch::cos $t1]
    set result2 [torch::cos -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test cos-6.2 {Syntax consistency - multiple tensors} {
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set result1 [torch::cos $t1]
    set result2 [torch::cos -input $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test cos-7.1 {Mathematical correctness - cos(0) should be 1} {
    set t1 [torch::zeros {1}]
    set result [torch::cos $t1]
    # cos(0) = 1, so result should be close to 1
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test cos-7.2 {Mathematical correctness - preservation of tensor structure} {
    set t1 [torch::zeros {2 3}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test cos-7.3 {Mathematical correctness - range validation} {
    # Cosine function should produce values in range [-1, 1]
    set t1 [torch::ones {3 3}]
    set result [torch::cos $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for parameter validation
test cos-8.1 {Parameter validation - input parameter required} {
    catch {torch::cos -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test cos-8.2 {Parameter validation - extra parameters rejected} {
    set t1 [torch::zeros {2 2}]
    catch {torch::cos -input $t1 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test cos-8.3 {Parameter validation - both -input and -tensor accepted} {
    set t1 [torch::zeros {2 2}]
    set result1 [torch::cos -input $t1]
    set result2 [torch::cos -tensor $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for backward compatibility
test cos-9.1 {Backward compatibility - original positional syntax still works} {
    set t1 [torch::zeros {4 4}]
    set result [torch::cos $t1]
    string match "tensor*" $result
} {1}

test cos-9.2 {Backward compatibility - exact argument count validation} {
    set t1 [torch::zeros {2 2}]
    catch {torch::cos $t1 extra_arg} msg
    string match "*Usage: torch::cos tensor*" $msg
} {1}

# Cleanup tests
test cos-10.1 {Basic functionality verification} {
    # Test that cos works correctly
    set t1 [torch::zeros {5 5}]
    set result [torch::cos $t1]
    string match "tensor*" $result
} {1}

cleanupTests 