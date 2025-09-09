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

# Test cases for basic functionality
test get_rng_state-1.1 {Basic functionality - returns tensor} {
    set state [torch::get_rng_state]
    ;# Should return a tensor handle
    string match "tensor*" $state
} {1}

test get_rng_state-1.2 {Returns valid tensor handle} {
    set state [torch::get_rng_state]
    ;# Should return non-empty string
    expr {[string length $state] > 0}
} {1}

test get_rng_state-1.3 {Tensor has correct properties} {
    set state [torch::get_rng_state]
    set shape [torch::tensor_shape $state]
    set dtype [torch::tensor_dtype $state]
    ;# Should be a 1D tensor with 64 elements of type Int64
    expr {$shape == 64 && $dtype eq "Int64"}
} {1}

test get_rng_state-1.4 {Consistent size across calls} {
    set state1 [torch::get_rng_state]
    set state2 [torch::get_rng_state]
    set shape1 [torch::tensor_shape $state1]
    set shape2 [torch::tensor_shape $state2]
    ;# Both should have same shape
    expr {$shape1 == $shape2}
} {1}

# Test cases for camelCase alias
test get_rng_state-2.1 {CamelCase alias functionality} {
    set state [torch::getRngState]
    ;# Should return a tensor handle
    string match "tensor*" $state
} {1}

test get_rng_state-2.2 {CamelCase alias returns valid tensor} {
    set state [torch::getRngState]
    ;# Should return non-empty string
    expr {[string length $state] > 0}
} {1}

test get_rng_state-2.3 {Both syntaxes return equivalent results} {
    set state1 [torch::get_rng_state]
    set state2 [torch::getRngState]
    set shape1 [torch::tensor_shape $state1]
    set shape2 [torch::tensor_shape $state2]
    set dtype1 [torch::tensor_dtype $state1]
    set dtype2 [torch::tensor_dtype $state2]
    ;# Both should have same shape and dtype
    expr {$shape1 == $shape2 && $dtype1 eq $dtype2}
} {1}

# Test cases for error handling
test get_rng_state-3.1 {No arguments accepted - snake_case} {
    set result [catch {torch::get_rng_state extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

test get_rng_state-3.2 {No arguments accepted - camelCase} {
    set result [catch {torch::getRngState extra_arg} msg]
    list $result [string match "*wrong # args*" $msg]
} {1 1}

# Test cases for tensor properties
test get_rng_state-4.1 {RNG state has correct data type} {
    set state [torch::get_rng_state]
    set dtype [torch::tensor_dtype $state]
    expr {$dtype eq "Int64"}
} {1}

test get_rng_state-4.2 {RNG state has correct size} {
    set state [torch::get_rng_state]
    set shape [torch::tensor_shape $state]
    ;# Should be 64 elements
    expr {$shape == 64}
} {1}

# Test cases for integration with set_rng_state
test get_rng_state-5.1 {Integration with set_rng_state} {
    ;# Get current state
    set state [torch::get_rng_state]
    
    ;# Set a new state (using the same state)
    torch::set_rng_state $state
    
    ;# Get state again
    set new_state [torch::get_rng_state]
    
    ;# Should have same properties
    set shape1 [torch::tensor_shape $state]
    set shape2 [torch::tensor_shape $new_state]
    expr {$shape1 == $shape2}
} {1}

test get_rng_state-5.2 {Multiple state operations} {
    ;# Get multiple states
    set state1 [torch::get_rng_state]
    set state2 [torch::get_rng_state]
    set state3 [torch::getRngState]
    
    ;# All should have same properties
    set shape1 [torch::tensor_shape $state1]
    set shape2 [torch::tensor_shape $state2]
    set shape3 [torch::tensor_shape $state3]
    
    expr {$shape1 == $shape2 && $shape2 == $shape3}
} {1}

# Test cases for memory management
test get_rng_state-6.1 {Multiple calls create different tensors} {
    set state1 [torch::get_rng_state]
    set state2 [torch::get_rng_state]
    
    ;# Should be different tensor handles
    expr {$state1 ne $state2}
} {1}

# Test cases for consistency
test get_rng_state-7.1 {Consistent tensor properties} {
    set states {}
    for {set i 0} {$i < 5} {incr i} {
        lappend states [torch::get_rng_state]
    }
    
    ;# All should have same shape and dtype
    set first_state [lindex $states 0]
    set first_shape [torch::tensor_shape $first_state]
    set first_dtype [torch::tensor_dtype $first_state]
    
    set all_consistent 1
    foreach state $states {
        set shape [torch::tensor_shape $state]
        set dtype [torch::tensor_dtype $state]
        if {$shape != $first_shape || $dtype ne $first_dtype} {
            set all_consistent 0
            break
        }
    }
    set all_consistent
} {1}

test get_rng_state-7.2 {CamelCase consistency} {
    set states_snake {}
    set states_camel {}
    
    for {set i 0} {$i < 3} {incr i} {
        lappend states_snake [torch::get_rng_state]
        lappend states_camel [torch::getRngState]
    }
    
    ;# All should have same properties
    set all_consistent 1
    foreach snake $states_snake camel $states_camel {
        set shape_snake [torch::tensor_shape $snake]
        set shape_camel [torch::tensor_shape $camel]
        set dtype_snake [torch::tensor_dtype $snake]
        set dtype_camel [torch::tensor_dtype $camel]
        
        if {$shape_snake != $shape_camel || $dtype_snake ne $dtype_camel} {
            set all_consistent 0
            break
        }
    }
    set all_consistent
} {1}

cleanupTests 