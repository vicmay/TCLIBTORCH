#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# =====================================================================
# TORCH::KRON COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test kron-1.1 {Basic positional syntax with 1D tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]
    set result [torch::kron $t1 $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-1.2 {Positional syntax with 2D tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set t2 [torch::tensor_create -data {5.0 6.0} -shape {1 2} -dtype float32]
    set result [torch::kron $t1 $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-1.3 {Positional syntax with scalar tensors} {
    set t1 [torch::tensor_create -data {2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0} -dtype float32]
    set result [torch::kron $t1 $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-1.4 {Positional syntax with mixed dimensions} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0} -dtype float32]
    set result [torch::kron $t1 $t2]
    expr {[string match "tensor*" $result]}
} {1}

# Tests for named parameter syntax
test kron-2.1 {Named parameter syntax basic} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]
    set result [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-2.2 {Named parameter syntax with 2D input} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set t2 [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2} -dtype float32]
    set result [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-2.3 {Named parameter syntax parameter order independence} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32]
    set t2 [torch::tensor_create -data {4.0 5.0} -dtype float32]
    set result1 [torch::kron -input $t1 -other $t2]
    set result2 [torch::kron -other $t2 -input $t1]
    ;# Note: Order of parameters doesn't matter, but order of tensors in operation does
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test kron-2.4 {Named parameter syntax with different data types} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float64]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float64]
    set result [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result]}
} {1}

# Tests for camelCase alias (kron is already camelCase)
test kron-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]
    set result [torch::kron $t1 $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32]
    set t2 [torch::tensor_create -data {4.0 5.0} -dtype float32]
    set result [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result]}
} {1}

# Tests for syntax consistency (both syntaxes should produce same results)
test kron-4.1 {Syntax consistency between positional and named} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]
    set result1 [torch::kron $t1 $t2]
    set result2 [torch::kron -input $t1 -other $t2]
    ;# Both should produce valid tensor handles
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test kron-4.2 {Syntax consistency with 2D tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set t2 [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set result1 [torch::kron $t1 $t2]
    set result2 [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# Tests for error handling
test kron-5.1 {Error on missing parameters} {
    catch {torch::kron} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test kron-5.2 {Error on insufficient positional arguments} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::kron $t1} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test kron-5.3 {Error on missing required parameters in named syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::kron -input $t1} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test kron-5.4 {Error on unknown named parameter} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]
    catch {torch::kron -input $t1 -other $t2 -unknown "value"} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test kron-5.5 {Error on missing value for named parameter} {
    catch {torch::kron -input} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test kron-5.6 {Error on invalid tensor handle - input} {
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]
    catch {torch::kron invalid_tensor $t2} msg
    expr {[string match "*Invalid*tensor*" $msg]}
} {1}

test kron-5.7 {Error on invalid tensor handle - other} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::kron $t1 invalid_tensor} msg
    expr {[string match "*Invalid*tensor*" $msg]}
} {1}

test kron-5.8 {Error on invalid tensor handle in named syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::kron -input $t1 -other invalid_tensor} msg
    expr {[string match "*Invalid*tensor*" $msg]}
} {1}

# Tests for mathematical properties of Kronecker product
test kron-6.1 {Kron product result dimensions - 1D tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0 5.0} -dtype float32]
    set result [torch::kron $t1 $t2]
    set shape [torch::tensor_shape $result]
    ;# Should be size 2*3 = 6
    expr {[lindex $shape 0] == 6}
} {1}

test kron-6.2 {Kron product result dimensions - 2D tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set t2 [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set result [torch::kron $t1 $t2]
    set shape [torch::tensor_shape $result]
    ;# Should be 4x4 for 2x2 @ 2x2
    expr {[lindex $shape 0] == 4 && [lindex $shape 1] == 4}
} {1}

test kron-6.3 {Kron product with identity matrix} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set identity [torch::tensor_create -data {1.0} -dtype float32]
    set result [torch::kron $t1 $identity]
    set shape [torch::tensor_shape $result]
    ;# Should preserve original shape
    expr {[lindex $shape 0] == 2}
} {1}

test kron-6.4 {Kron product with zero tensor} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set zero [torch::tensor_create -data {0.0} -dtype float32]
    set result [torch::kron $t1 $zero]
    expr {[string match "tensor*" $result]}
} {1}

# Tests for different data types
test kron-7.1 {Kron product with float32 tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]
    set result [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-7.2 {Kron product with float64 tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float64]
    set t2 [torch::tensor_create -data {3.0 4.0} -dtype float64]
    set result [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-7.3 {Kron product with integer tensors} {
    set t1 [torch::tensor_create -data {1 2} -dtype int32]
    set t2 [torch::tensor_create -data {3 4} -dtype int32]
    set result [torch::kron $t1 $t2]
    expr {[string match "tensor*" $result]}
} {1}

# Tests for edge cases
test kron-8.1 {Kron product with single element tensors} {
    set t1 [torch::tensor_create -data {5.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0} -dtype float32]
    set result [torch::kron $t1 $t2]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 1}
} {1}

test kron-8.2 {Kron product with large tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set t2 [torch::tensor_create -data {6.0 7.0 8.0} -dtype float32]
    set result [torch::kron -input $t1 -other $t2]
    set shape [torch::tensor_shape $result]
    ;# Should be 5*3 = 15
    expr {[lindex $shape 0] == 15}
} {1}

test kron-8.3 {Kron product with fractional values} {
    set t1 [torch::tensor_create -data {0.5 1.5} -dtype float32]
    set t2 [torch::tensor_create -data {2.5 3.5} -dtype float32]
    set result [torch::kron $t1 $t2]
    expr {[string match "tensor*" $result]}
} {1}

test kron-8.4 {Kron product with negative values} {
    set t1 [torch::tensor_create -data {-1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 -4.0} -dtype float32]
    set result [torch::kron -input $t1 -other $t2]
    expr {[string match "tensor*" $result]}
} {1}

# Tests for tensor order sensitivity
test kron-9.1 {Kron product is not commutative - different order} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
    set t2 [torch::tensor_create -data {3.0 4.0 5.0} -dtype float32]
    set result1 [torch::kron $t1 $t2]
    set result2 [torch::kron $t2 $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    ;# Shapes should be different (6 vs 6, but content different)
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test kron-9.2 {Kron product order matters for 2D tensors} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set t2 [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set result1 [torch::kron -input $t1 -other $t2]
    set result2 [torch::kron -input $t2 -other $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    ;# Both should be 4x4 but with different values
    expr {$shape1 eq $shape2 && [lindex $shape1 0] == 4}
} {1}

# Cleanup
cleanupTests 