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
test tensor-unique-1.1 {Basic positional syntax - unique values only} {
    set t1 [torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0 4.0} -dtype float32]
    set result [torch::tensor_unique $t1]
    set unique_values [torch::tensor_to_list -input $result]
    return $unique_values
} {1.0 2.0 3.0 4.0}

test tensor-unique-1.2 {Positional syntax with sorted=false} {
    set t2 [torch::tensor_create -data {3.0 1.0 2.0 1.0 4.0 2.0} -dtype float32]
    set result [torch::tensor_unique $t2 0]
    set unique_values [torch::tensor_to_list -input $result]
    return $unique_values
} {1.0 2.0 3.0 4.0}

test tensor-unique-1.3 {Positional syntax with return_inverse=true} {
    set t3 [torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0} -dtype float32]
    set result [torch::tensor_unique $t3 1 1]
    regexp {unique (\S+) inverse (\S+)} $result -> unique_name inverse_name
    regsub {\}$} $inverse_name {} inverse_name
    set unique_values [torch::tensor_to_list -input $unique_name]
    set inverse_indices [torch::tensor_to_list -input $inverse_name]
    return [list $unique_values $inverse_indices]
} {{1.0 2.0 3.0} {0 1 0 2 1}}

# Test cases for named parameter syntax
test tensor-unique-2.1 {Named parameter syntax - unique values only} {
    set t4 [torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0 4.0} -dtype float32]
    set result [torch::tensor_unique -tensor $t4]
    set unique_values [torch::tensor_to_list -input $result]
    return $unique_values
} {1.0 2.0 3.0 4.0}

test tensor-unique-2.2 {Named parameter syntax with sorted=false} {
    set t5 [torch::tensor_create -data {3.0 1.0 2.0 1.0 4.0 2.0} -dtype float32]
    set result [torch::tensor_unique -tensor $t5 -sorted 0]
    set unique_values [torch::tensor_to_list -input $result]
    return $unique_values
} {1.0 2.0 3.0 4.0}

test tensor-unique-2.3 {Named parameter syntax with returnInverse=true} {
    set t6 [torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0} -dtype float32]
    set result [torch::tensor_unique -tensor $t6 -sorted 1 -returnInverse 1]
    regexp {unique (\S+) inverse (\S+)} $result -> unique_name inverse_name
    regsub {\}$} $inverse_name {} inverse_name
    set unique_values [torch::tensor_to_list -input $unique_name]
    set inverse_indices [torch::tensor_to_list -input $inverse_name]
    return [list $unique_values $inverse_indices]
} {{1.0 2.0 3.0} {0 1 0 2 1}}

# Test cases for camelCase alias
test tensor-unique-3.1 {CamelCase alias - unique values only} {
    set t7 [torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0 4.0} -dtype float32]
    set result [torch::tensorUnique -tensor $t7]
    set unique_values [torch::tensor_to_list -input $result]
    return $unique_values
} {1.0 2.0 3.0 4.0}

test tensor-unique-3.2 {CamelCase alias with returnInverse} {
    set t8 [torch::tensor_create -data {1.0 2.0 1.0 3.0 2.0} -dtype float32]
    set result [torch::tensorUnique -tensor $t8 -returnInverse 1]
    regexp {unique (\S+) inverse (\S+)} $result -> unique_name inverse_name
    regsub {\}$} $inverse_name {} inverse_name
    set unique_values [torch::tensor_to_list -input $unique_name]
    set inverse_indices [torch::tensor_to_list -input $inverse_name]
    return [list $unique_values $inverse_indices]
} {{1.0 2.0 3.0} {0 1 0 2 1}}

# Error handling tests
test tensor-unique-4.1 {Error handling - missing tensor} {
    catch {torch::tensor_unique} result
    return $result
} {wrong # args: should be "torch::tensor_unique tensor ?sorted? ?return_inverse? OR -tensor tensor -sorted bool -returnInverse bool"}

test tensor-unique-4.2 {Error handling - tensor not found} {
    catch {torch::tensor_unique nonexistent_tensor} result
    return $result
} {Tensor not found}

test tensor-unique-4.3 {Error handling - invalid parameter} {
    set t9 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    catch {torch::tensor_unique -tensor $t9 -invalid 1} result
    return $result
} {Unknown parameter: -invalid}

test tensor-unique-4.4 {Error handling - missing parameter value} {
    set t10 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    catch {torch::tensor_unique -tensor $t10 -sorted} result
    return $result
} {Missing value for parameter}

# Edge cases
test tensor-unique-5.1 {Edge case - single value} {
    set t11 [torch::tensor_create -data {5.0} -dtype float32]
    set result [torch::tensor_unique $t11]
    set unique_values [torch::tensor_to_list -input $result]
    return $unique_values
} {5.0}

test tensor-unique-5.2 {Edge case - all unique values} {
    set t12 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set result [torch::tensor_unique $t12]
    set unique_values [torch::tensor_to_list -input $result]
    return $unique_values
} {1.0 2.0 3.0 4.0 5.0}

cleanupTests 