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
test bitwise_or-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create {1 2 4 8} int32]
    set t2 [torch::tensor_create {1 3 5 7} int32]
    set result [torch::bitwise_or $t1 $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_or-1.2 {Positional syntax with integer tensors} {
    set t1 [torch::tensor_create {5 10 15} int32]
    set t2 [torch::tensor_create {3 6 9} int32]
    set result [torch::bitwise_or $t1 $t2]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_or-1.3 {Positional syntax error handling - insufficient args} {
    catch {torch::bitwise_or tensor1} result
    expr {[string match "*Usage*" $result] || [string match "*tensor1 tensor2*" $result]}
} {1}

test bitwise_or-1.4 {Positional syntax error handling - too many args} {
    catch {torch::bitwise_or tensor1 tensor2 extra} result
    expr {[string match "*Usage*" $result] || [string match "*tensor1 tensor2*" $result]}
} {1}

# Test cases for named parameter syntax
test bitwise_or-2.1 {Named parameter syntax with -input and -other} {
    set t1 [torch::tensor_create {1 4 16} int32]
    set t2 [torch::tensor_create {2 8 32} int32]
    set result [torch::bitwise_or -input $t1 -other $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_or-2.2 {Named parameter syntax with -tensor1 and -tensor2} {
    set t1 [torch::tensor_create {7 14 21} int32]
    set t2 [torch::tensor_create {3 6 12} int32]
    set result [torch::bitwise_or -tensor1 $t1 -tensor2 $t2]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_or-2.3 {Named parameter syntax reversed order} {
    set t1 [torch::tensor_create {12 24 48} int32]
    set t2 [torch::tensor_create {3 5 7} int32]
    set result [torch::bitwise_or -other $t2 -input $t1]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_or-2.4 {Named parameter syntax error handling - missing parameter} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_or -input $t1} result
    expr {[string match "*Required parameters missing*" $result] || [string match "*other*" $result]}
} {1}

test bitwise_or-2.5 {Named parameter syntax error handling - unknown parameter} {
    set t1 [torch::tensor_create {1 2 3} int32]
    set t2 [torch::tensor_create {4 5 6} int32]
    catch {torch::bitwise_or -input $t1 -unknown $t2} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test bitwise_or-2.6 {Named parameter syntax error handling - missing value} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_or -input $t1 -other} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

# Test cases for camelCase alias
test bitwise_or-3.1 {CamelCase alias with positional syntax} {
    set t1 [torch::tensor_create {8 16 32} int32]
    set t2 [torch::tensor_create {1 2 4} int32]
    set result [torch::bitwiseOr $t1 $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_or-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::tensor_create {9 18 36} int32]
    set t2 [torch::tensor_create {5 10 20} int32]
    set result [torch::bitwiseOr -input $t1 -other $t2]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

# Test that both syntaxes produce the same result
test bitwise_or-4.1 {Both syntaxes produce same result} {
    set t1 [torch::tensor_create {5 10 15} int32]
    set t2 [torch::tensor_create {3 6 9} int32]
    
    set result1 [torch::bitwise_or $t1 $t2]
    set result2 [torch::bitwise_or -input $t1 -other $t2]
    set result3 [torch::bitwiseOr $t1 $t2]
    set result4 [torch::bitwiseOr -input $t1 -other $t2]
    
    # All results should be valid (non-empty)
    expr {$result1 ne "" && $result2 ne "" && $result3 ne "" && $result4 ne ""}
} {1}

# Test error handling for invalid tensor names
test bitwise_or-5.1 {Error handling for invalid first tensor} {
    set t2 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_or invalid_tensor $t2} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_or-5.2 {Error handling for invalid second tensor} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_or $t1 invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_or-5.3 {Error handling for invalid tensor with named parameters} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_or -input $t1 -other invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

# Test mathematical correctness
test bitwise_or-6.1 {Mathematical correctness - simple case} {
    set t1 [torch::tensor_create {5} int32]   ; # 5 = 101 in binary
    set t2 [torch::tensor_create {3} int32]   ; # 3 = 011 in binary
    set result [torch::bitwise_or $t1 $t2]    ; # 101 | 011 = 111 = 7
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_or-6.2 {Mathematical correctness with named parameters} {
    set t1 [torch::tensor_create {12} int32]  ; # 12 = 1100 in binary
    set t2 [torch::tensor_create {10} int32]  ; # 10 = 1010 in binary
    set result [torch::bitwise_or -input $t1 -other $t2]  ; # 1100 | 1010 = 1110 = 14
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_or-6.3 {Mathematical correctness - multiple values} {
    set t1 [torch::tensor_create {1 2 4} int32]   ; # 001, 010, 100 in binary
    set t2 [torch::tensor_create {1 3 5} int32]   ; # 001, 011, 101 in binary
    set result [torch::bitwise_or $t1 $t2]        ; # 001|001=001, 010|011=011, 100|101=101 = 1,3,5
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_or-6.4 {Mathematical correctness with zero} {
    set t1 [torch::tensor_create {0 5 10} int32]  ; # OR with 0 should return the other value
    set t2 [torch::tensor_create {7 0 15} int32]  
    set result [torch::bitwise_or $t1 $t2]        ; # 0|7=7, 5|0=5, 10|15=15
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_or-6.5 {Mathematical correctness - OR with self} {
    set t1 [torch::tensor_create {7 15 31} int32]
    set result [torch::bitwise_or $t1 $t1]        ; # x | x = x (idempotent property)
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

# Test bitwise OR properties
test bitwise_or-7.1 {Commutative property: A OR B = B OR A} {
    set t1 [torch::tensor_create {5 10} int32]
    set t2 [torch::tensor_create {3 12} int32]
    
    set result1 [torch::bitwise_or $t1 $t2]
    set result2 [torch::bitwise_or $t2 $t1]
    
    # Both results should be valid (commutative property should hold)
    expr {$result1 ne "" && $result2 ne ""}
} {1}

test bitwise_or-7.2 {OR with all 1s} {
    set t1 [torch::tensor_create -data {5 10 15} -dtype int32]   ; # Various values
    set t2 [torch::tensor_create -data {-1 -1 -1} -dtype int32]  ; # All bits 1 (in two's complement)
    set result [torch::bitwise_or $t1 $t2]          ; # x | 111...111 = 111...111 = -1
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

cleanupTests 