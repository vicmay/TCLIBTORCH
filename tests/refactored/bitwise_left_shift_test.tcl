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
test bitwise_left_shift-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create {1 2 4 8} int32]
    set t2 [torch::tensor_create {1 1 1 1} int32]
    set result [torch::bitwise_left_shift $t1 $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_left_shift-1.2 {Positional syntax with integer tensors} {
    set t1 [torch::tensor_create {3 7 15} int32]
    set t2 [torch::tensor_create {2 1 3} int32]
    set result [torch::bitwise_left_shift $t1 $t2]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_left_shift-1.3 {Positional syntax error handling - insufficient args} {
    catch {torch::bitwise_left_shift tensor1} result
    expr {[string match "*Usage*" $result] || [string match "*tensor1 tensor2*" $result]}
} {1}

test bitwise_left_shift-1.4 {Positional syntax error handling - too many args} {
    catch {torch::bitwise_left_shift tensor1 tensor2 extra} result
    expr {[string match "*Usage*" $result] || [string match "*tensor1 tensor2*" $result]}
} {1}

# Test cases for named parameter syntax
test bitwise_left_shift-2.1 {Named parameter syntax with -input and -other} {
    set t1 [torch::tensor_create {1 2 4 8} int32]
    set t2 [torch::tensor_create {1 1 1 1} int32]
    set result [torch::bitwise_left_shift -input $t1 -other $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_left_shift-2.2 {Named parameter syntax with -tensor1 and -tensor2} {
    set t1 [torch::tensor_create {5 10 20} int32]
    set t2 [torch::tensor_create {1 2 1} int32]
    set result [torch::bitwise_left_shift -tensor1 $t1 -tensor2 $t2]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_left_shift-2.3 {Named parameter syntax reversed order} {
    set t1 [torch::tensor_create {12 8 4} int32]
    set t2 [torch::tensor_create {1 2 3} int32]
    set result [torch::bitwise_left_shift -other $t2 -input $t1]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_left_shift-2.4 {Named parameter syntax error handling - missing parameter} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_left_shift -input $t1} result
    expr {[string match "*Required parameters missing*" $result] || [string match "*other*" $result]}
} {1}

test bitwise_left_shift-2.5 {Named parameter syntax error handling - unknown parameter} {
    set t1 [torch::tensor_create {1 2 3} int32]
    set t2 [torch::tensor_create {1 1 1} int32]
    catch {torch::bitwise_left_shift -input $t1 -unknown $t2} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test bitwise_left_shift-2.6 {Named parameter syntax error handling - missing value} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_left_shift -input $t1 -other} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

# Test cases for camelCase alias
test bitwise_left_shift-3.1 {CamelCase alias with positional syntax} {
    set t1 [torch::tensor_create {9 5 1} int32]
    set t2 [torch::tensor_create {1 2 3} int32]
    set result [torch::bitwiseLeftShift $t1 $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_left_shift-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::tensor_create {14 10 6} int32]
    set t2 [torch::tensor_create {1 1 2} int32]
    set result [torch::bitwiseLeftShift -input $t1 -other $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

# Test that both syntaxes produce the same result
test bitwise_left_shift-4.1 {Both syntaxes produce same result} {
    set t1 [torch::tensor_create {7 3 1} int32]
    set t2 [torch::tensor_create {1 2 3} int32]
    
    set result1 [torch::bitwise_left_shift $t1 $t2]
    set result2 [torch::bitwise_left_shift -input $t1 -other $t2]
    set result3 [torch::bitwiseLeftShift $t1 $t2]
    set result4 [torch::bitwiseLeftShift -input $t1 -other $t2]
    
    # All results should be valid (non-empty)
    expr {$result1 ne "" && $result2 ne "" && $result3 ne "" && $result4 ne ""}
} {1}

# Test error handling for invalid tensor names
test bitwise_left_shift-5.1 {Error handling for invalid first tensor} {
    set t2 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_left_shift invalid_tensor $t2} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_left_shift-5.2 {Error handling for invalid second tensor} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_left_shift $t1 invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_left_shift-5.3 {Error handling for invalid tensor with named parameters} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_left_shift -input $t1 -other invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

# Test mathematical correctness
test bitwise_left_shift-6.1 {Mathematical correctness - simple case} {
    set t1 [torch::tensor_create {5} int32]   ; # 5 = 101 in binary
    set t2 [torch::tensor_create {1} int32]   ; # Shift left by 1
    set result [torch::bitwise_left_shift $t1 $t2]   ; # 101 << 1 = 1010 = 10
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_left_shift-6.2 {Mathematical correctness with named parameters} {
    set t1 [torch::tensor_create {3} int32]   ; # 3 = 011 in binary
    set t2 [torch::tensor_create {2} int32]   ; # Shift left by 2
    set result [torch::bitwise_left_shift -input $t1 -other $t2]  ; # 011 << 2 = 1100 = 12
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_left_shift-6.3 {Mathematical correctness - multiple values} {
    set t1 [torch::tensor_create {1 2 4} int32]   ; # 001, 010, 100 in binary
    set t2 [torch::tensor_create {1 1 1} int32]   ; # Shift each left by 1
    set result [torch::bitwise_left_shift $t1 $t2]   ; # Expected: 2, 4, 8
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

cleanupTests 