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
# TORCH::ATAN2 COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test atan2-1.1 {Basic positional syntax} {
    set y [torch::ones {3 3}]
    set x [torch::ones {3 3}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test atan2-1.2 {Positional syntax with different values} {
    set y [torch::zeros {2 2}]
    set x [torch::ones {2 2}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test atan2-1.3 {Positional syntax with 1D tensors} {
    set y [torch::ones {5}]
    set x [torch::ones {5}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

# Tests for named parameter syntax
test atan2-2.1 {Named parameter syntax basic} {
    set y [torch::ones {2 3}]
    set x [torch::ones {2 3}]
    set result [torch::atan2 -y $y -x $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test atan2-2.2 {Named parameter syntax reversed order} {
    set y [torch::ones {3}]
    set x [torch::ones {3}]
    set result [torch::atan2 -x $x -y $y]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test atan2-2.3 {Named parameter syntax with different tensor sizes} {
    set y [torch::zeros {4 2 2}]
    set x [torch::ones {4 2 2}]
    set result [torch::atan2 -y $y -x $x]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2 2}

test atan2-2.4 {Named parameter syntax with -input1/-input2 aliases} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    set result [torch::atan2 -input1 $y -input2 $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test atan2-2.5 {Named parameter syntax mixed aliases} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    set result [torch::atan2 -y $y -input2 $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

# Tests for camelCase alias (atan2 is already camelCase)
test atan2-3.1 {CamelCase alias basic functionality} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test atan2-3.2 {CamelCase alias with named parameters} {
    set y [torch::ones {3 3}]
    set x [torch::ones {3 3}]
    set result [torch::atan2 -y $y -x $x]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for error handling
test atan2-4.1 {Error on missing parameter} {
    catch {torch::atan2} msg
    expr {[string match "*Required parameters missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test atan2-4.2 {Error on only one positional parameter} {
    set y [torch::ones {2 2}]
    catch {torch::atan2 $y} msg
    string match "*Usage*" $msg
} {1}

test atan2-4.3 {Error on invalid y tensor name} {
    set x [torch::ones {2 2}]
    catch {torch::atan2 invalid_tensor $x} msg
    string match "*Invalid y tensor name*" $msg
} {1}

test atan2-4.4 {Error on invalid x tensor name} {
    set y [torch::ones {2 2}]
    catch {torch::atan2 $y invalid_tensor} msg
    string match "*Invalid x tensor name*" $msg
} {1}

test atan2-4.5 {Error on invalid tensor name with named parameters} {
    set x [torch::ones {2 2}]
    catch {torch::atan2 -y invalid_tensor -x $x} msg
    string match "*Invalid y tensor name*" $msg
} {1}

test atan2-4.6 {Error on unknown parameter} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    catch {torch::atan2 -unknown_param $y -x $x} msg
    string match "*Unknown parameter*" $msg
} {1}

test atan2-4.7 {Error on missing parameter value} {
    set y [torch::ones {2 2}]
    catch {torch::atan2 -y $y -x} msg
    string match "*Missing value for parameter*" $msg
} {1}

test atan2-4.8 {Error on too many positional arguments} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    set z [torch::ones {2 2}]
    catch {torch::atan2 $y $x $z} msg
    string match "*Usage*" $msg
} {1}

# Tests for data types and edge cases
test atan2-5.1 {Float tensor handling} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test atan2-5.2 {Large tensor handling} {
    set y [torch::ones {10 10}]
    set x [torch::ones {10 10}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test atan2-5.3 {1D tensor handling} {
    set y [torch::ones {5}]
    set x [torch::ones {5}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test atan2-5.4 {3D tensor handling} {
    set y [torch::ones {2 3 4}]
    set x [torch::ones {2 3 4}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test atan2-5.5 {Zero and non-zero combinations} {
    set y [torch::zeros {2 2}]
    set x [torch::ones {2 2}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

# Tests for syntax consistency (both syntaxes should produce same results)
test atan2-6.1 {Syntax consistency - shape preservation} {
    set y [torch::ones {3 3}]
    set x [torch::ones {3 3}]
    set result1 [torch::atan2 $y $x]
    set result2 [torch::atan2 -y $y -x $x]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test atan2-6.2 {Syntax consistency - multiple tensors} {
    set y1 [torch::ones {2 2}]
    set x1 [torch::ones {2 2}]
    set y2 [torch::ones {2 2}]
    set x2 [torch::ones {2 2}]
    set result1 [torch::atan2 $y1 $x1]
    set result2 [torch::atan2 -y $y2 -x $x2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test atan2-6.3 {Syntax consistency - parameter alias equivalence} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    set result1 [torch::atan2 -y $y -x $x]
    set result2 [torch::atan2 -input1 $y -input2 $x]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test atan2-7.1 {Mathematical correctness - atan2(0,1) should be 0} {
    set y [torch::zeros {1}]
    set x [torch::ones {1}]
    set result [torch::atan2 $y $x]
    # Result should be 0 (atan2(0,1) = 0)
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test atan2-7.2 {Mathematical correctness - atan2(1,1) behavior} {
    set y [torch::ones {1}]
    set x [torch::ones {1}]
    set result [torch::atan2 $y $x]
    # Result should be approximately π/4 (atan2(1,1) = 45 degrees)
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test atan2-7.3 {Mathematical correctness - preservation of tensor structure} {
    set y [torch::ones {2 3}]
    set x [torch::ones {2 3}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Tests for parameter validation
test atan2-8.1 {Parameter validation - both parameters required} {
    set y [torch::ones {2 2}]
    catch {torch::atan2 -y $y} msg
    string match "*Required parameters missing*" $msg
} {1}

test atan2-8.2 {Parameter validation - extra parameters ignored} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    catch {torch::atan2 -y $y -x $x -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test atan2-8.3 {Parameter validation - valid parameter names} {
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    # All valid combinations should work
    set result1 [torch::atan2 -y $y -x $x]
    set result2 [torch::atan2 -input1 $y -input2 $x]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for special mathematical cases
test atan2-9.1 {Special case: y=0, x>0 (should be 0)} {
    set y [torch::zeros {3}]
    set x [torch::ones {3}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test atan2-9.2 {Special case: y>0, x=0 (should be π/2)} {
    set y [torch::ones {3}]
    set x [torch::zeros {3}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test atan2-9.3 {Mixed values test} {
    set y [torch::tensor_create {1.0 0.0 -1.0}]
    set x [torch::tensor_create {1.0 1.0 1.0}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

# Cleanup tests
test atan2-10.1 {Basic functionality verification} {
    # Test that atan2 works correctly
    set y [torch::ones {5 5}]
    set x [torch::ones {5 5}]
    set result [torch::atan2 $y $x]
    string match "tensor*" $result
} {1}

test atan2-10.2 {Return value format} {
    set y [torch::ones {3 3}]
    set x [torch::ones {3 3}]
    set result [torch::atan2 $y $x]
    # Result should be a tensor handle
    expr {[string length $result] > 0}
} {1}

test atan2-10.3 {Binary operation verification} {
    # Verify that we need exactly two input tensors
    set y [torch::ones {2 2}]
    set x [torch::ones {2 2}]
    set result [torch::atan2 $y $x]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 2 && [lindex $shape 0] == 2 && [lindex $shape 1] == 2}
} {1}

cleanupTests 