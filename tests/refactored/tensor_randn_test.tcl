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
test tensor-randn-1.1 {Basic positional syntax - 1D tensor} {
    set t [torch::tensor_randn {5}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "5"}
} {1}

test tensor-randn-1.2 {Positional syntax - 2D tensor, cpu, float32} {
    set t [torch::tensor_randn {2 3} cpu float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 3"}
} {1}

test tensor-randn-1.3 {Positional syntax - 3D tensor, cpu, float64} {
    set t [torch::tensor_randn {2 2 2} cpu float64]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2 2"}
} {1}

# Test cases for named parameter syntax
test tensor-randn-2.1 {Named parameter syntax - 1D tensor} {
    set t [torch::tensor_randn -shape {7}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "7"}
} {1}

test tensor-randn-2.2 {Named parameter syntax - 2D tensor, cpu, float64} {
    set t [torch::tensor_randn -shape {3 4} -device cpu -dtype float64]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "3 4"}
} {1}

test tensor-randn-2.3 {Named parameter syntax - 3D tensor, float32} {
    set t [torch::tensor_randn -shape {2 2 2} -dtype float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2 2"}
} {1}

# Test cases for camelCase alias
test tensor-randn-3.1 {CamelCase alias - 1D tensor} {
    set t [torch::tensorRandn {4}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "4"}
} {1}

test tensor-randn-3.2 {CamelCase alias - named parameters} {
    set t [torch::tensorRandn -shape {2 2} -dtype float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2"}
} {1}

# Error handling tests
test tensor-randn-4.1 {Error - missing shape} {
    catch {torch::tensor_randn} result
    set result
} {Required parameter missing: shape}

test tensor-randn-4.2 {Error - unknown parameter} {
    catch {torch::tensor_randn -foo {2 2}} result
    set result
} {Unknown parameter: -foo}

test tensor-randn-4.3 {Error - missing value for parameter} {
    catch {torch::tensor_randn -shape} result
    set result
} {Missing value for parameter}

test tensor-randn-4.4 {Error - too many positional arguments} {
    catch {torch::tensor_randn {2 2} cpu float32 extra} result
    set result
} {Invalid number of arguments}

# Edge cases
test tensor-randn-5.1 {Edge case - empty shape (scalar)} {
    set t [torch::tensor_randn {}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq ""}
} {1}

test tensor-randn-5.2 {Edge case - large tensor} {
    set t [torch::tensor_randn {10 10}]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "10 10"}
} {1}

# Device and dtype support
test tensor-randn-6.1 {Device - cpu, dtype - float32} {
    set t [torch::tensor_randn -shape {2 2} -device cpu -dtype float32]
    set shape [torch::tensor_shape $t]
    expr {$shape eq "2 2"}
} {1}

# Syntax consistency
test tensor-randn-7.1 {Syntax consistency - positional vs named} {
    set t1 [torch::tensor_randn {2 2}]
    set t2 [torch::tensor_randn -shape {2 2}]
    set s1 [torch::tensor_shape $t1]
    set s2 [torch::tensor_shape $t2]
    expr {$s1 eq $s2}
} {1}

test tensor-randn-7.2 {Syntax consistency - snake_case vs camelCase} {
    set t1 [torch::tensor_randn {2 2}]
    set t2 [torch::tensorRandn {2 2}]
    set s1 [torch::tensor_shape $t1]
    set s2 [torch::tensor_shape $t2]
    expr {$s1 eq $s2}
} {1}

# Mathematical correctness
test tensor-randn-8.1 {Mathematical correctness - normal distribution properties} {
    set t [torch::tensor_randn {1000}]
    set vals [torch::tensor_print $t]
    # Check that we have some negative and positive values (normal distribution)
    # The output contains newlines and tensor type info, so we need to parse carefully
    set has_negative 0
    set has_positive 0
    foreach line [split $vals "\n"] {
        # Skip empty lines and tensor type info
        if {[string trim $line] eq ""} continue
        if {[string first "CPU" $line] >= 0} continue
        if {[string first "Type" $line] >= 0} continue
        
        # Parse each value in the line
        foreach v [split [string trim $line] " "] {
            if {[string is double -strict $v]} {
                if {$v < 0} {set has_negative 1}
                if {$v > 0} {set has_positive 1}
            }
        }
    }
    expr {$has_negative && $has_positive}
} {1}

cleanupTests 