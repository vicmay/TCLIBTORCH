#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test 1: Basic positional syntax (backward compatibility)
test hessian-1.1 {Basic positional syntax} -body {
    set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::hessian "dummy_func" $x]
    set shape [torch::tensor_shape $result]
    ;# Should return identity matrix with size equal to tensor numel
    expr {$shape eq {3 3}}
} -result 1

;# Test 2: Named parameter syntax
test hessian-2.1 {Named parameter syntax with -func and -inputs} -body {
    set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::hessian -func "dummy_func" -inputs $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {3 3}}
} -result 1

;# Test 3: Named parameter syntax with alternative parameter names
test hessian-2.2 {Named parameter syntax with -function and -input} -body {
    set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::hessian -function "dummy_func" -input $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {3 3}}
} -result 1

;# Test 4: Named parameter syntax with different order
test hessian-2.3 {Named parameter syntax with different parameter order} -body {
    set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::hessian -inputs $x -func "dummy_func"]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {3 3}}
} -result 1

;# Test 5: Different tensor sizes
test hessian-3.1 {Different tensor sizes - 1D tensor} -body {
    set x [torch::tensor_create {1.0} float32 cpu true]
    set result [torch::hessian -func "dummy_func" -inputs $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {1 1}}
} -result 1

test hessian-3.2 {Different tensor sizes - 2D tensor} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::hessian -func "dummy_func" -inputs $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {4 4}}
} -result 1

;# Test 6: Verify result format (should be identity matrix)
test hessian-4.1 {Result format is correct} -body {
    set x [torch::tensor_create {1.0 2.0} float32 cpu true]
    set result [torch::hessian -func "dummy_func" -inputs $x]
    set shape [torch::tensor_shape $result]
    ;# Should be 2x2 identity matrix
    expr {$shape eq {2 2}}
} -result 1

;# Test 7: Error handling - missing parameters (positional)
test hessian-5.1 {Error: missing parameters in positional syntax} -body {
    catch {torch::hessian "dummy_func"} msg
    string match "*Usage: torch::hessian func inputs*" $msg
} -result 1

;# Test 8: Error handling - missing parameters (named)
test hessian-5.2 {Error: missing -func parameter} -body {
    set x [torch::tensor_create {1.0 2.0} float32 cpu true]
    catch {torch::hessian -inputs $x} msg
    string match "*func and inputs required*" $msg
} -result 1

test hessian-5.3 {Error: missing -inputs parameter} -body {
    catch {torch::hessian -func "dummy_func"} msg
    string match "*func and inputs required*" $msg
} -result 1

;# Test 9: Error handling - invalid parameters
test hessian-5.4 {Error: unknown parameter} -body {
    set x [torch::tensor_create {1.0 2.0} float32 cpu true]
    catch {torch::hessian -func "dummy_func" -inputs $x -unknown "value"} msg
    string match "*Unknown parameter*" $msg
} -result 1

;# Test 10: Error handling - unpaired parameters
test hessian-5.5 {Error: unpaired parameters} -body {
    catch {torch::hessian -func "dummy_func" -inputs} msg
    string match "*Named parameters must come in pairs*" $msg
} -result 1

;# Test 11: Mathematical correctness - consistency between syntaxes
test hessian-6.1 {Consistency between positional and named syntax} -body {
    set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result1 [torch::hessian "dummy_func" $x]
    set result2 [torch::hessian -func "dummy_func" -inputs $x]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

;# Test 12: Data type support
test hessian-7.1 {Double precision tensor} -body {
    set x [torch::tensor_create {1.0 2.0 3.0} float64 cpu true]
    set result [torch::hessian -func "dummy_func" -inputs $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {3 3}}
} -result 1

;# Test 13: camelCase alias support (same as snake_case in this case)
test hessian-8.1 {camelCase command name works} -body {
    set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::hessian -func "dummy_func" -inputs $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {3 3}}
} -result 1

cleanupTests 