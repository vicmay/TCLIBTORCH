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

# Test cases for positional syntax
test tensor-rand-1.1 {Basic positional syntax - 1D tensor} {
    set t [torch::tensor_rand {5}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "5"}
} {1}

test tensor-rand-1.2 {Positional syntax - 2D tensor, cpu, float32} {
    set t [torch::tensor_rand {2 3} cpu float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 3"}
} {1}

test tensor-rand-1.3 {Positional syntax - 3D tensor, cpu, float64} {
    set t [torch::tensor_rand {2 2 2} cpu float64]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2 2"}
} {1}

# Test cases for named parameter syntax
test tensor-rand-2.1 {Named parameter syntax - 1D tensor} {
    set t [torch::tensor_rand -shape {7}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "7"}
} {1}

test tensor-rand-2.2 {Named parameter syntax - 2D tensor, cpu, float64} {
    set t [torch::tensor_rand -shape {3 4} -device cpu -dtype float64]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "3 4"}
} {1}

test tensor-rand-2.3 {Named parameter syntax - 3D tensor, float32} {
    set t [torch::tensor_rand -shape {2 2 2} -dtype float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2 2"}
} {1}

# Test cases for camelCase alias
test tensor-rand-3.1 {CamelCase alias - 1D tensor} {
    set t [torch::tensorRand {4}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "4"}
} {1}

test tensor-rand-3.2 {CamelCase alias - named parameters} {
    set t [torch::tensorRand -shape {2 2} -dtype float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2"}
} {1}

# Error handling tests
test tensor-rand-4.1 {Error - missing shape} {
    catch {torch::tensor_rand} result
    set result
} {Required parameter missing: shape}

test tensor-rand-4.2 {Error - unknown parameter} {
    catch {torch::tensor_rand -foo {2 2}} result
    set result
} {Unknown parameter: -foo}

test tensor-rand-4.3 {Error - missing value for parameter} {
    catch {torch::tensor_rand -shape} result
    set result
} {Missing value for parameter}

test tensor-rand-4.4 {Error - too many positional arguments} {
    catch {torch::tensor_rand {2 2} cpu float32 extra} result
    set result
} {Invalid number of arguments}

# Edge cases
test tensor-rand-5.1 {Edge case - empty shape (scalar)} {
    set t [torch::tensor_rand {}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq ""}
} {1}

test tensor-rand-5.2 {Edge case - large tensor} {
    set t [torch::tensor_rand {10 10}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "10 10"}
} {1}

# Device and dtype support
test tensor-rand-6.1 {Device - cpu, dtype - float32} {
    set t [torch::tensor_rand -shape {2 2} -device cpu -dtype float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2"}
} {1}

# Syntax consistency
test tensor-rand-7.1 {Syntax consistency - positional vs named} {
    set t1 [torch::tensor_rand {2 2}]
    set t2 [torch::tensor_rand -shape {2 2}]
    set s1 [torch::tensor_shape $t1]
    set s2 [torch::tensor_shape $t2]
    expr {$s1 eq $s2}
} {1}

test tensor-rand-7.2 {Syntax consistency - snake_case vs camelCase} {
    set t1 [torch::tensor_rand {2 2}]
    set t2 [torch::tensorRand {2 2}]
    set s1 [torch::tensor_shape $t1]
    set s2 [torch::tensor_shape $t2]
    expr {$s1 eq $s2}
} {1}

# Mathematical correctness
test tensor-rand-8.1 {Mathematical correctness - values in [0,1)} {
    set t [torch::tensor_rand {5}]
    set vals [torch::tensor_print $t]
    set ok 1
    foreach v [split $vals " "] {
        if {[string is double -strict $v]} {
            if {![expr {$v >= 0.0 && $v < 1.0}]} {set ok 0}
        }
    }
    set ok
} {1}

cleanupTests 