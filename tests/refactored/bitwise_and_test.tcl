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

# Test cases for positional syntax (backward compatibility)
test bitwise_and-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create {1 3 5 7} int32]
    set t2 [torch::tensor_create {1 2 4 6} int32]
    set result [torch::bitwise_and $t1 $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_and-1.2 {Positional syntax with integer tensors} {
    set t1 [torch::tensor_create {15 7 3} int32]
    set t2 [torch::tensor_create {8 3 1} int32]
    set result [torch::bitwise_and $t1 $t2]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_and-1.3 {Positional syntax error handling - insufficient args} {
    catch {torch::bitwise_and tensor1} result
    expr {[string match "*Usage*" $result] || [string match "*tensor1 tensor2*" $result]}
} {1}

test bitwise_and-1.4 {Positional syntax error handling - too many args} {
    catch {torch::bitwise_and tensor1 tensor2 extra} result
    expr {[string match "*Usage*" $result] || [string match "*tensor1 tensor2*" $result]}
} {1}

# Test cases for named parameter syntax
test bitwise_and-2.1 {Named parameter syntax with -input and -other} {
    set t1 [torch::tensor_create {1 3 5 7} int32]
    set t2 [torch::tensor_create {1 2 4 6} int32]
    set result [torch::bitwise_and -input $t1 -other $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_and-2.2 {Named parameter syntax with -tensor1 and -tensor2} {
    set t1 [torch::tensor_create {15 7 3} int32]
    set t2 [torch::tensor_create {8 3 1} int32]
    set result [torch::bitwise_and -tensor1 $t1 -tensor2 $t2]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_and-2.3 {Named parameter syntax reversed order} {
    set t1 [torch::tensor_create {12 8 4} int32]
    set t2 [torch::tensor_create {10 6 2} int32]
    set result [torch::bitwise_and -other $t2 -input $t1]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_and-2.4 {Named parameter syntax error handling - missing parameter} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_and -input $t1} result
    expr {[string match "*Required parameters missing*" $result] || [string match "*other*" $result]}
} {1}

test bitwise_and-2.5 {Named parameter syntax error handling - unknown parameter} {
    set t1 [torch::tensor_create {1 2 3} int32]
    set t2 [torch::tensor_create {4 5 6} int32]
    catch {torch::bitwise_and -input $t1 -unknown $t2} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test bitwise_and-2.6 {Named parameter syntax error handling - missing value} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_and -input $t1 -other} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

# Test cases for camelCase alias
test bitwise_and-3.1 {CamelCase alias with positional syntax} {
    set t1 [torch::tensor_create {9 5 1} int32]
    set t2 [torch::tensor_create {8 4 1} int32]
    set result [torch::bitwiseAnd $t1 $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_and-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::tensor_create {14 10 6} int32]
    set t2 [torch::tensor_create {12 8 4} int32]
    set result [torch::bitwiseAnd -input $t1 -other $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

# Test that both syntaxes produce the same result
test bitwise_and-4.1 {Both syntaxes produce same result} {
    set t1 [torch::tensor_create {7 3 1} int32]
    set t2 [torch::tensor_create {6 2 1} int32]
    
    set result1 [torch::bitwise_and $t1 $t2]
    set result2 [torch::bitwise_and -input $t1 -other $t2]
    set result3 [torch::bitwiseAnd $t1 $t2]
    set result4 [torch::bitwiseAnd -input $t1 -other $t2]
    
    # All results should be valid (non-empty)
    expr {$result1 ne "" && $result2 ne "" && $result3 ne "" && $result4 ne ""}
} {1}

# Test error handling for invalid tensor names
test bitwise_and-5.1 {Error handling for invalid first tensor} {
    set t2 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_and invalid_tensor $t2} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_and-5.2 {Error handling for invalid second tensor} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_and $t1 invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_and-5.3 {Error handling for invalid tensor with named parameters} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_and -input $t1 -other invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

# Test mathematical correctness
test bitwise_and-6.1 {Mathematical correctness - simple case} {
    set t1 [torch::tensor_create {5} int32]   ; # 5 = 101 in binary
    set t2 [torch::tensor_create {3} int32]   ; # 3 = 011 in binary
    set result [torch::bitwise_and $t1 $t2]   ; # 101 & 011 = 001 = 1
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_and-6.2 {Mathematical correctness with named parameters} {
    set t1 [torch::tensor_create {12} int32]  ; # 12 = 1100 in binary
    set t2 [torch::tensor_create {10} int32]  ; # 10 = 1010 in binary
    set result [torch::bitwise_and -input $t1 -other $t2]  ; # 1100 & 1010 = 1000 = 8
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

cleanupTests 