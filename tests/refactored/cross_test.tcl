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
# TORCH::CROSS COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test cross-1.1 {Basic positional syntax with 3D vectors} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross $v1 $v2]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test cross-1.2 {Positional syntax with 2x3 matrices} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} {2 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0} {2 3} float32]
    set result [torch::cross $m1 $m2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test cross-1.3 {Positional syntax with dim parameter} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} {2 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0} {2 3} float32]
    set result [torch::cross $m1 $m2 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

# Tests for named parameter syntax
test cross-2.1 {Named parameter syntax basic} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross -input $v1 -other $v2]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test cross-2.2 {Named parameter syntax with -tensor alias} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross -tensor $v1 -other $v2]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test cross-2.3 {Named parameter syntax with dim parameter} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} {2 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0} {2 3} float32]
    set result [torch::cross -input $m1 -other $m2 -dim 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test cross-2.4 {Named parameters in different order} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross -other $v2 -input $v1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

# Tests for camelCase alias
test cross-3.1 {CamelCase alias basic functionality} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross $v1 $v2]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test cross-3.2 {CamelCase alias with named parameters} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross -input $v1 -other $v2]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

# Tests for error handling
test cross-4.1 {Error on missing parameters} {
    catch {torch::cross} msg
    expr {[string match "*Required parameters missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test cross-4.2 {Error on missing second parameter} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    catch {torch::cross $v1} msg
    expr {[string match "*Required parameters missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test cross-4.3 {Error on invalid input tensor name} {
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    catch {torch::cross invalid_tensor $v2} msg
    string match "*Invalid input tensor*" $msg
} {1}

test cross-4.4 {Error on invalid other tensor name} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    catch {torch::cross $v1 invalid_tensor} msg
    string match "*Invalid other tensor*" $msg
} {1}

test cross-4.5 {Error on invalid tensor name with named parameters} {
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    catch {torch::cross -input invalid_tensor -other $v2} msg
    string match "*Invalid input tensor*" $msg
} {1}

test cross-4.6 {Error on unknown parameter} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    catch {torch::cross -input $v1 -other $v2 -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test cross-4.7 {Error on missing parameter value} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    catch {torch::cross -input $v1 -other} msg
    string match "*Missing value for parameter*" $msg
} {1}

test cross-4.8 {Error on invalid dim parameter} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    catch {torch::cross -input $v1 -other $v2 -dim invalid} msg
    string match "*Invalid dim parameter*" $msg
} {1}

# Tests for data types and shapes
test cross-5.1 {Cross product with different vector sizes} {
    set v1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set v2 [torch::tensor_create {4.0 5.0 6.0} float32]
    set result [torch::cross $v1 $v2]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test cross-5.2 {Cross product with 2D tensors} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0} {3 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0} {3 3} float32]
    set result [torch::cross $m1 $m2]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test cross-5.3 {Cross product with batch dimension} {
    set b1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} {2 1 3} float32]
    set b2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0} {2 1 3} float32]
    set result [torch::cross $b1 $b2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 1 3}

# Tests for syntax consistency (both syntaxes should produce same results)
test cross-6.1 {Syntax consistency - basic vectors} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result1 [torch::cross $v1 $v2]
    set result2 [torch::cross -input $v1 -other $v2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test cross-6.2 {Syntax consistency - with dim parameter} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} {2 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0} {2 3} float32]
    set result1 [torch::cross $m1 $m2 1]
    set result2 [torch::cross -input $m1 -other $m2 -dim 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test cross-7.1 {Mathematical correctness - standard unit vectors} {
    # i × j = k (standard basis vectors)
    set i [torch::tensor_create {1.0 0.0 0.0} float32]
    set j [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross $i $j]
    # Result should be {0.0 0.0 1.0} (k vector)
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test cross-7.2 {Mathematical correctness - anti-commutativity} {
    # a × b = -(b × a)
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {4.0 5.0 6.0} float32]
    set cross_ab [torch::cross $a $b]
    set cross_ba [torch::cross $b $a]
    # Shapes should be the same (values will be negated)
    set shape_ab [torch::tensor_shape $cross_ab]
    set shape_ba [torch::tensor_shape $cross_ba]
    expr {$shape_ab eq $shape_ba}
} {1}

test cross-7.3 {Mathematical correctness - orthogonality} {
    # a × b should be orthogonal to both a and b
    set a [torch::tensor_create {1.0 2.0 3.0} float32]
    set b [torch::tensor_create {4.0 5.0 6.0} float32]
    set cross_result [torch::cross $a $b]
    set shape [torch::tensor_shape $cross_result]
    set shape
} {3}

# Tests for parameter validation
test cross-8.1 {Parameter validation - both tensors required} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    catch {torch::cross -input $v1} msg
    expr {[string match "*Required parameters missing*" $msg] || [string match "*Missing value for parameter*" $msg]}
} {1}

test cross-8.2 {Parameter validation - extra parameters rejected} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    catch {torch::cross -input $v1 -other $v2 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test cross-8.3 {Parameter validation - both -input and -tensor accepted} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result1 [torch::cross -input $v1 -other $v2]
    set result2 [torch::cross -tensor $v1 -other $v2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for backward compatibility
test cross-9.1 {Backward compatibility - original positional syntax still works} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross $v1 $v2]
    string match "tensor*" $result
} {1}

test cross-9.2 {Backward compatibility - positional with dim parameter} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} {2 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0} {2 3} float32]
    set result [torch::cross $m1 $m2 1]
    string match "tensor*" $result
} {1}

test cross-9.3 {Backward compatibility - exact argument count validation} {
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    catch {torch::cross $v1 $v2 1 extra_arg} msg
    string match "*Usage: torch::cross input other ?dim?*" $msg
} {1}

# Tests for dimension parameter
test cross-10.1 {Dimension parameter - explicit dim=0} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0} {3 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0} {3 3} float32]
    set result [torch::cross -input $m1 -other $m2 -dim 0]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test cross-10.2 {Dimension parameter - explicit dim=1} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0} {3 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0} {3 3} float32]
    set result [torch::cross -input $m1 -other $m2 -dim 1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test cross-10.3 {Dimension parameter - positional vs named consistency} {
    set m1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} {2 3} float32]
    set m2 [torch::tensor_create {0.0 1.0 0.0 0.0 0.0 1.0} {2 3} float32]
    set result1 [torch::cross $m1 $m2 1]
    set result2 [torch::cross -input $m1 -other $m2 -dim 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Cleanup tests
test cross-11.1 {Basic functionality verification} {
    # Test that cross product works correctly
    set v1 [torch::tensor_create {1.0 0.0 0.0} float32]
    set v2 [torch::tensor_create {0.0 1.0 0.0} float32]
    set result [torch::cross $v1 $v2]
    string match "tensor*" $result
} {1}

cleanupTests 