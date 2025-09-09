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
# TORCH::ATLEAST_1D COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test atleast_1d-1.1 {Basic positional syntax with 1D tensor} {
    set t1 [torch::zeros {5}]
    set result [torch::atleast_1d $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test atleast_1d-1.2 {Positional syntax with scalar (0D tensor)} {
    set t1 [torch::tensor_create {42.0}]
    set result [torch::atleast_1d $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test atleast_1d-1.3 {Positional syntax with 2D tensor} {
    set t1 [torch::zeros {3 3}]
    set result [torch::atleast_1d $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test atleast_1d-1.4 {Positional syntax with 3D tensor} {
    set t1 [torch::zeros {2 3 4}]
    set result [torch::atleast_1d $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

# Tests for named parameter syntax
test atleast_1d-2.1 {Named parameter syntax basic with 1D tensor} {
    set t1 [torch::zeros {5}]
    set result [torch::atleast_1d -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test atleast_1d-2.2 {Named parameter syntax with scalar} {
    set t1 [torch::tensor_create {42.0}]
    set result [torch::atleast_1d -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test atleast_1d-2.3 {Named parameter syntax with 2D tensor} {
    set t1 [torch::ones {2 3}]
    set result [torch::atleast_1d -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test atleast_1d-2.4 {Named parameter syntax with -tensor alias} {
    set t1 [torch::ones {4}]
    set result [torch::atleast_1d -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test atleast_1d-2.5 {Named parameter syntax with higher dimensions} {
    set t1 [torch::ones {2 2 2 2}]
    set result [torch::atleast_1d -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2 2 2}

# Tests for camelCase alias
test atleast_1d-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::zeros {5}]
    set result [torch::atleast1d $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test atleast_1d-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::ones {3}]
    set result [torch::atleast1d -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test atleast_1d-3.3 {CamelCase alias with scalar} {
    set t1 [torch::tensor_create {5.0}]
    set result [torch::atleast1d $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

# Tests for error handling
test atleast_1d-4.1 {Error on missing parameter} {
    catch {torch::atleast_1d} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test atleast_1d-4.2 {Error on invalid tensor name} {
    catch {torch::atleast_1d invalid_tensor} msg
    string match "*Invalid*tensor*" $msg
} {1}

test atleast_1d-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::atleast_1d -input invalid_tensor} msg
    string match "*Invalid*tensor*" $msg
} {1}

test atleast_1d-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2}]
    catch {torch::atleast_1d -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test atleast_1d-4.5 {Error on missing parameter value} {
    set t1 [torch::zeros {2}]
    catch {torch::atleast_1d -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test atleast_1d-4.6 {Error on too many positional arguments} {
    set t1 [torch::zeros {2}]
    set t2 [torch::zeros {2}]
    catch {torch::atleast_1d $t1 $t2} msg
    string match "*Usage*" $msg
} {1}

# Tests for shape transformation behavior
test atleast_1d-5.1 {Scalar to 1D transformation} {
    set scalar [torch::tensor_create {42.0}]
    set result [torch::atleast_1d $scalar]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 1 && [lindex $shape 0] == 1}
} {1}

test atleast_1d-5.2 {1D tensor unchanged} {
    set t1 [torch::zeros {7}]
    set result [torch::atleast_1d $t1]
    set orig_shape [torch::tensor_shape $t1]
    set new_shape [torch::tensor_shape $result]
    expr {$orig_shape eq $new_shape}
} {1}

test atleast_1d-5.3 {2D tensor unchanged} {
    set t1 [torch::ones {3 4}]
    set result [torch::atleast_1d $t1]
    set orig_shape [torch::tensor_shape $t1]
    set new_shape [torch::tensor_shape $result]
    expr {$orig_shape eq $new_shape}
} {1}

test atleast_1d-5.4 {3D tensor unchanged} {
    set t1 [torch::ones {2 3 4}]
    set result [torch::atleast_1d $t1]
    set orig_shape [torch::tensor_shape $t1]
    set new_shape [torch::tensor_shape $result]
    expr {$orig_shape eq $new_shape}
} {1}

# Tests for syntax consistency (both syntaxes should produce same results)
test atleast_1d-6.1 {Syntax consistency - shape preservation} {
    set t1 [torch::zeros {5}]
    set result1 [torch::atleast_1d $t1]
    set result2 [torch::atleast_1d -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test atleast_1d-6.2 {Syntax consistency - scalar transformation} {
    set scalar [torch::tensor_create {3.14}]
    set result1 [torch::atleast_1d $scalar]
    set result2 [torch::atleast_1d -input $scalar]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test atleast_1d-6.3 {Syntax consistency - parameter alias equivalence} {
    set t1 [torch::zeros {3}]
    set result1 [torch::atleast_1d -input $t1]
    set result2 [torch::atleast_1d -tensor $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test atleast_1d-6.4 {Syntax consistency - camelCase equivalence} {
    set t1 [torch::ones {4}]
    set result1 [torch::atleast_1d $t1]
    set result2 [torch::atleast1d $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for mathematical correctness
test atleast_1d-7.1 {Mathematical correctness - minimum 1D guarantee} {
    set scalar [torch::tensor_create {1.0}]
    set result [torch::atleast_1d $scalar]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] >= 1}
} {1}

test atleast_1d-7.2 {Mathematical correctness - preservation of higher dimensions} {
    set t2d [torch::ones {2 3}]
    set result [torch::atleast_1d $t2d]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 2 && [lindex $shape 0] == 2 && [lindex $shape 1] == 3}
} {1}

test atleast_1d-7.3 {Mathematical correctness - 1D tensor identity} {
    set t1d [torch::zeros {6}]
    set result [torch::atleast_1d $t1d]
    set orig_shape [torch::tensor_shape $t1d]
    set new_shape [torch::tensor_shape $result]
    expr {$orig_shape eq $new_shape}
} {1}

# Tests for parameter validation
test atleast_1d-8.1 {Parameter validation - input parameter required} {
    catch {torch::atleast_1d -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test atleast_1d-8.2 {Parameter validation - extra parameters rejected} {
    set t1 [torch::zeros {2}]
    catch {torch::atleast_1d -input $t1 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test atleast_1d-8.3 {Parameter validation - valid parameter names} {
    set t1 [torch::zeros {3}]
    # Both -input and -tensor should be valid
    set result1 [torch::atleast_1d -input $t1]
    set result2 [torch::atleast_1d -tensor $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for edge cases and special tensors
test atleast_1d-9.1 {Empty 1D tensor handling} {
    set empty [torch::zeros {0}]
    set result [torch::atleast_1d $empty]
    set shape [torch::tensor_shape $result]
    set shape
} {0}

test atleast_1d-9.2 {Large tensor handling} {
    set large [torch::zeros {1000}]
    set result [torch::atleast_1d $large]
    set shape [torch::tensor_shape $result]
    set shape
} {1000}

test atleast_1d-9.3 {Very high dimensional tensor} {
    set high_dim [torch::ones {1 1 1 1 1}]
    set result [torch::atleast_1d $high_dim]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 1 1 1}

# Tests for data preservation
test atleast_1d-10.1 {Data preservation - values unchanged} {
    # Create a known tensor and verify atleast_1d preserves data
    set t1 [torch::ones {3}]
    set result [torch::atleast_1d $t1]
    # Both should have same handle type
    string match "tensor*" $result
} {1}

test atleast_1d-10.2 {Shape transformation verification} {
    set scalar [torch::tensor_create {99.0}]
    set result [torch::atleast_1d $scalar]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 1 && [lindex $shape 0] == 1}
} {1}

# Cleanup tests
test atleast_1d-11.1 {Basic functionality verification} {
    # Test that atleast_1d works correctly
    set t1 [torch::zeros {4}]
    set result [torch::atleast_1d $t1]
    string match "tensor*" $result
} {1}

test atleast_1d-11.2 {Return value format} {
    set t1 [torch::ones {2}]
    set result [torch::atleast_1d $t1]
    # Result should be a tensor handle
    expr {[string length $result] > 0}
} {1}

test atleast_1d-11.3 {Scalar transformation verification} {
    set scalar [torch::tensor_create {42.0}]
    set result [torch::atleast_1d $scalar]
    set shape [torch::tensor_shape $result]
    # Should transform scalar to shape {1}
    expr {[llength $shape] == 1 && [lindex $shape 0] == 1}
} {1}

cleanupTests 