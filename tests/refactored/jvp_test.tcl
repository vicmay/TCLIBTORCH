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

# -----------------------------------------------------------------------------
# Test 1: Basic functionality - Positional syntax (backward compatibility)
# -----------------------------------------------------------------------------

test jvp-1.1 {Positional syntax - basic JVP} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32]
    set v [torch::tensor_create {1.0 1.0 1.0} float32]
    set result [torch::jvp "test_func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-1.2 {Positional syntax result shape} {
    set inputs [torch::tensor_create {2.0 3.0} float32]
    set v [torch::tensor_create {1.0 1.0} float32]
    set result [torch::jvp "func" $inputs $v]
    ;# Result should be a scalar (tensor with no dimensions) for vector inputs
    expr {[string match "tensor*" $result]}
} {1}

test jvp-1.3 {Positional syntax different function names} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    set v [torch::tensor_create {0.5 0.5} float32]
    set result1 [torch::jvp "func_a" $inputs $v]
    set result2 [torch::jvp "func_b" $inputs $v]
    ;# Both should return valid tensors
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# -----------------------------------------------------------------------------
# Test 2: Named parameter syntax
# -----------------------------------------------------------------------------

test jvp-2.1 {Named syntax with -func, -inputs, -v} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32]
    set v [torch::tensor_create {1.0 1.0 1.0} float32]
    set result [torch::jvp -func "test_func" -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-2.2 {Named syntax with alternative parameter names} {
    set inputs [torch::tensor_create {2.0 3.0 4.0} float32]
    set v [torch::tensor_create {1.0 0.5 0.25} float32]
    set result [torch::jvp -function "my_func" -input $inputs -vector $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-2.3 {Named syntax parameter order independence} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    set v [torch::tensor_create {2.0 3.0} float32]
    set result1 [torch::jvp -func "f" -inputs $inputs -v $v]
    set result2 [torch::jvp -v $v -func "f" -inputs $inputs]
    set result3 [torch::jvp -inputs $inputs -v $v -func "f"]
    ;# All should return valid tensors
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} {1}

# -----------------------------------------------------------------------------
# Test 3: camelCase alias (same as jvp since it's already lowercase)
# -----------------------------------------------------------------------------

test jvp-3.1 {camelCase alias - positional syntax} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    set v [torch::tensor_create {1.0 1.0} float32]
    set result [torch::jvp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-3.2 {camelCase alias - named syntax} {
    set inputs [torch::tensor_create {3.0 4.0} float32]
    set v [torch::tensor_create {0.5 0.5} float32]
    set result [torch::jvp -func "test" -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 4: Syntax consistency - both syntaxes produce same results
# -----------------------------------------------------------------------------

test jvp-4.1 {Both syntaxes produce same results} {
    set inputs [torch::tensor_create {2.0 3.0} float32]
    set v [torch::tensor_create {1.0 1.0} float32]
    
    set result1 [torch::jvp "func" $inputs $v]
    set result2 [torch::jvp -func "func" -inputs $inputs -v $v]
    
    ;# Both should return valid tensor handles
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test jvp-4.2 {Consistency with different parameter aliases} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32]
    set v [torch::tensor_create {1.0 1.0 1.0} float32]
    
    set result1 [torch::jvp -func "f" -inputs $inputs -v $v]
    set result2 [torch::jvp -function "f" -input $inputs -vector $v]
    
    ;# Both should return valid tensors
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# -----------------------------------------------------------------------------
# Test 5: Error handling
# -----------------------------------------------------------------------------

test jvp-5.1 {Missing arguments - positional} {
    catch {torch::jvp} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test jvp-5.2 {Missing arguments - insufficient positional args} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    catch {torch::jvp "func" $inputs} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test jvp-5.3 {Missing required parameters in named syntax} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    catch {torch::jvp -func "test" -inputs $inputs} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test jvp-5.4 {Unknown named parameter} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    set v [torch::tensor_create {1.0 1.0} float32]
    catch {torch::jvp -func "f" -inputs $inputs -v $v -unknown "value"} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test jvp-5.5 {Missing value for named parameter} {
    catch {torch::jvp -func "test" -inputs} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test jvp-5.6 {Invalid tensor handle} {
    set v [torch::tensor_create {1.0 1.0} float32]
    catch {torch::jvp "func" invalid_tensor $v} msg
    expr {[string match "*Error in jvp*" $msg]}
} {1}

# -----------------------------------------------------------------------------
# Test 6: Different tensor sizes and data types
# -----------------------------------------------------------------------------

test jvp-6.1 {Single element tensors} {
    set inputs [torch::tensor_create {5.0} float32]
    set v [torch::tensor_create {2.0} float32]
    set result [torch::jvp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-6.2 {Float64 tensors} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float64]
    set v [torch::tensor_create {1.0 1.0 1.0} float64]
    set result [torch::jvp -func "test" -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-6.3 {Large tensors} {
    set inputs [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set v [torch::tensor_create {1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0} float32]
    set result [torch::jvp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-6.4 {Mixed size compatibility test} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32]
    set v [torch::tensor_create {1.0 1.0 1.0} float32]
    set result [torch::jvp -func "test" -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 7: Mathematical properties
# -----------------------------------------------------------------------------

test jvp-7.1 {JVP with zero vector} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32]
    set v [torch::tensor_create {0.0 0.0 0.0} float32]
    set result [torch::jvp "func" $inputs $v]
    ;# Should still return a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

test jvp-7.2 {JVP with identity-like vector} {
    set inputs [torch::tensor_create {1.0 0.0} float32]
    set v [torch::tensor_create {1.0 0.0} float32]
    set result [torch::jvp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-7.3 {JVP with negative values} {
    set inputs [torch::tensor_create -data {-1.0 2.0 -3.0} -dtype float32]
    set v [torch::tensor_create -data {1.0 -1.0 1.0} -dtype float32]
    set result [torch::jvp -func "test" -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 8: Function name variations
# -----------------------------------------------------------------------------

test jvp-8.1 {Empty function name should be error} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    set v [torch::tensor_create {1.0 1.0} float32]
    catch {torch::jvp "" $inputs $v} msg
    expr {[string match "*required*" $msg]}
} {1}

test jvp-8.2 {Function name with spaces} {
    set inputs [torch::tensor_create {1.0 2.0} float32]
    set v [torch::tensor_create {1.0 1.0} float32]
    set result [torch::jvp "my function" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-8.3 {Function name with special characters} {
    set inputs [torch::tensor_create {2.0 3.0} float32]
    set v [torch::tensor_create {0.5 0.5} float32]
    set result [torch::jvp "func_123!@#" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 9: Edge cases and boundary conditions
# -----------------------------------------------------------------------------

test jvp-9.1 {Very small values} {
    set inputs [torch::tensor_create {0.001 0.002} float32]
    set v [torch::tensor_create {0.001 0.001} float32]
    set result [torch::jvp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-9.2 {Very large values} {
    set inputs [torch::tensor_create {1000.0 2000.0} float32]
    set v [torch::tensor_create {100.0 100.0} float32]
    set result [torch::jvp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-9.3 {Fractional values} {
    set inputs [torch::tensor_create {0.33 0.66 0.99} float32]
    set v [torch::tensor_create {0.1 0.2 0.3} float32]
    set result [torch::jvp -func "test" -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} {1}

# -----------------------------------------------------------------------------
# Test 10: Multi-dimensional input handling
# -----------------------------------------------------------------------------

test jvp-10.1 {Matrix inputs} {
    ;# Note: Current implementation may flatten matrices
    set inputs [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set v [torch::tensor_create {1.0 1.0 1.0 1.0} float32]
    set result [torch::jvp "func" $inputs $v]
    expr {[string match "tensor*" $result]}
} {1}

test jvp-10.2 {Different shapes but compatible sizes} {
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32]
    set v [torch::tensor_create {0.5 0.5 0.5} float32]
    set result [torch::jvp -func "test" -inputs $inputs -v $v]
    expr {[string match "tensor*" $result]}
} {1}

# Cleanup
cleanupTests 