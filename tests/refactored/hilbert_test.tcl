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
test hilbert-1.1 {Basic positional syntax} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::hilbert $x]
    ;# Check that result is a valid tensor handle
    expr {$result ne ""}
} -result 1

;# Test 2: Named parameter syntax with -input
test hilbert-2.1 {Named parameter syntax with -input} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::hilbert -input $x]
    expr {$result ne ""}
} -result 1

;# Test 3: Named parameter syntax with -tensor
test hilbert-2.2 {Named parameter syntax with -tensor} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::hilbert -tensor $x]
    expr {$result ne ""}
} -result 1

;# Test 4: Result shape verification
test hilbert-3.1 {Result shape matches input shape} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::hilbert -input $x]
    set input_shape [torch::tensor_shape $x]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} -result 1

;# Test 5: Different input sizes
test hilbert-4.1 {Different input sizes - small tensor} -body {
    set x [torch::tensor_create {1.0 2.0} float32 cpu true]
    set result [torch::hilbert -input $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {2}}
} -result 1

test hilbert-4.2 {Different input sizes - larger tensor} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]
    set result [torch::hilbert -input $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {8}}
} -result 1

;# Test 6: Mathematical properties - consistency between syntaxes
test hilbert-5.1 {Consistency between positional and named syntax} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result1 [torch::hilbert $x]
    set result2 [torch::hilbert -input $x]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

;# Test 7: Data type support
test hilbert-6.1 {Double precision tensor} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float64 cpu true]
    set result [torch::hilbert -input $x]
    expr {$result ne ""}
} -result 1

test hilbert-6.2 {Float32 tensor} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set result [torch::hilbert -input $x]
    expr {$result ne ""}
} -result 1

;# Test 8: Error handling - missing parameters (positional)
test hilbert-7.1 {Error: missing parameters in positional syntax} -body {
    catch {torch::hilbert} msg
    string match "*input tensor required*" $msg
} -result 1

;# Test 9: Error handling - missing parameters (named)
test hilbert-7.2 {Error: missing input parameter} -body {
    catch {torch::hilbert -input} msg
    string match "*Named parameters must come in pairs*" $msg
} -result 1

test hilbert-7.3 {Error: no parameters at all} -body {
    catch {torch::hilbert} msg
    string match "*input tensor required*" $msg
} -result 1

;# Test 10: Error handling - invalid parameters
test hilbert-7.4 {Error: unknown parameter} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    catch {torch::hilbert -input $x -unknown "value"} msg
    string match "*Unknown parameter*" $msg
} -result 1

;# Test 11: Error handling - invalid tensor handle
test hilbert-7.5 {Error: invalid tensor handle} -body {
    catch {torch::hilbert -input "invalid_handle"} msg
    string match "*Error in hilbert*" $msg
} -result 1

;# Test 12: Edge cases - single element tensor
test hilbert-8.1 {Single element tensor} -body {
    set x [torch::tensor_create {1.0} float32 cpu true]
    set result [torch::hilbert -input $x]
    set shape [torch::tensor_shape $result]
    expr {$shape eq {1}}
} -result 1

;# Test 13: Complex signal processing use case
test hilbert-9.1 {Real signal processing example} -body {
    ;# Create a simple sinusoidal signal
    set x [torch::tensor_create {0.0 1.0 0.0 -1.0 0.0 1.0 0.0 -1.0} float32 cpu true]
    set result [torch::hilbert -input $x]
    ;# Check that the result has the same length
    set shape [torch::tensor_shape $result]
    expr {$shape eq {8}}
} -result 1

;# Test 14: camelCase alias support (according to SQLite, hilbert has camelCase alias)
test hilbert-10.1 {camelCase command name works} -body {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    ;# The command should already support camelCase (same as snake_case in this case)
    set result [torch::hilbert -input $x]
    expr {$result ne ""}
} -result 1

cleanupTests 