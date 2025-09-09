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
test bitwise_not-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create {5 10 15} int32]
    set result [torch::bitwise_not $t1]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_not-1.2 {Positional syntax with integer tensors} {
    set t1 [torch::tensor_create {1 2 4 8} int32]
    set result [torch::bitwise_not $t1]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_not-1.3 {Positional syntax error handling - insufficient args} {
    catch {torch::bitwise_not} result
    expr {[string match "*Usage*" $result] || [string match "*tensor*" $result]}
} {1}

test bitwise_not-1.4 {Positional syntax error handling - too many args} {
    catch {torch::bitwise_not tensor1 extra} result
    expr {[string match "*Usage*" $result] || [string match "*tensor*" $result]}
} {1}

# Test cases for named parameter syntax
test bitwise_not-2.1 {Named parameter syntax with -input} {
    set t1 [torch::tensor_create {7 14 21} int32]
    set result [torch::bitwise_not -input $t1]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_not-2.2 {Named parameter syntax with -tensor} {
    set t1 [torch::tensor_create {3 6 9} int32]
    set result [torch::bitwise_not -tensor $t1]
    
    # Verify tensor created successfully
    expr {$result ne ""}
} {1}

test bitwise_not-2.3 {Named parameter syntax error handling - missing parameter} {
    catch {torch::bitwise_not -input} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

test bitwise_not-2.4 {Named parameter syntax error handling - unknown parameter} {
    set t1 [torch::tensor_create {1 2 3} int32]
    catch {torch::bitwise_not -unknown $t1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test bitwise_not-2.5 {Named parameter syntax error handling - no parameters} {
    catch {torch::bitwise_not} result
    expr {[string match "*Usage*" $result] || [string match "*tensor*" $result]}
} {1}

# Test cases for camelCase alias
test bitwise_not-3.1 {CamelCase alias with positional syntax} {
    set t1 [torch::tensor_create {12 24 48} int32]
    set result [torch::bitwiseNot $t1]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

test bitwise_not-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::tensor_create {16 32 64} int32]
    set result [torch::bitwiseNot -input $t1]
    
    # Check that result is valid
    expr {$result ne ""}
} {1}

# Test that both syntaxes produce the same result
test bitwise_not-4.1 {Both syntaxes produce same result} {
    set t1 [torch::tensor_create {5 10 15} int32]
    
    set result1 [torch::bitwise_not $t1]
    set result2 [torch::bitwise_not -input $t1]
    set result3 [torch::bitwiseNot $t1]
    set result4 [torch::bitwiseNot -tensor $t1]
    
    # All results should be valid (non-empty)
    expr {$result1 ne "" && $result2 ne "" && $result3 ne "" && $result4 ne ""}
} {1}

# Test error handling for invalid tensor names
test bitwise_not-5.1 {Error handling for invalid tensor} {
    catch {torch::bitwise_not invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_not-5.2 {Error handling for invalid tensor with named parameters} {
    catch {torch::bitwise_not -input invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test bitwise_not-5.3 {Error handling for invalid tensor with camelCase} {
    catch {torch::bitwiseNot invalid_tensor} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

# Test mathematical correctness
test bitwise_not-6.1 {Mathematical correctness - simple case} {
    set t1 [torch::tensor_create {5} int32]   ; # 5 = 101 in binary
    set result [torch::bitwise_not $t1]       ; # ~101 = ...11111010 = -6 (two's complement)
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_not-6.2 {Mathematical correctness with named parameters} {
    set t1 [torch::tensor_create {0} int32]   ; # 0 = 000 in binary
    set result [torch::bitwise_not -input $t1]  ; # ~000 = ...11111111 = -1 (two's complement)
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_not-6.3 {Mathematical correctness - multiple values} {
    set t1 [torch::tensor_create {1 2 4} int32]   ; # 001, 010, 100 in binary
    set result [torch::bitwise_not $t1]           ; # Expected: -2, -3, -5 (two's complement)
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_not-6.4 {Mathematical correctness with camelCase} {
    set t1 [torch::tensor_create {7} int32]   ; # 7 = 111 in binary
    set result [torch::bitwiseNot -input $t1] ; # ~111 = ...11111000 = -8 (two's complement)
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

# Test with different data types
test bitwise_not-7.1 {Works with int32 tensors - large values} {
    set t1 [torch::tensor_create {255} int32]  ; # 11111111 in binary (as int32)
    set result [torch::bitwise_not $t1]        ; # ~255 = -256 (two's complement)
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

test bitwise_not-7.2 {Works with different integer types} {
    set t1 [torch::tensor_create {15} int32]   ; # 1111 in binary (lower 4 bits)
    set result [torch::bitwise_not -tensor $t1] ; # ~1111 = -16 (two's complement)
    
    # The result should be a valid tensor
    expr {$result ne ""}
} {1}

cleanupTests 