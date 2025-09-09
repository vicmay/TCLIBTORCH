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

;# Test cases for positional syntax
test tensor_split-1.1 {Basic positional syntax - split by sections} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_split $tensor 3]
    set num_parts [llength $result]
    return $num_parts
} {3}

test tensor_split-1.2 {Basic positional syntax - split by indices} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_split $tensor {2 4}]
    set num_parts [llength $result]
    return $num_parts
} {3}

test tensor_split-1.3 {Positional syntax with dimension} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}}]
    set result [torch::tensor_split $tensor 2 0]
    set num_parts [llength $result]
    return $num_parts
} {2}

;# Test cases for named parameter syntax
test tensor_split-2.1 {Named parameter syntax - input and sections} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_split -input $tensor -sections 3]
    set num_parts [llength $result]
    return $num_parts
} {3}

test tensor_split-2.2 {Named parameter syntax - tensor and indices} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_split -tensor $tensor -indices {2 4}]
    set num_parts [llength $result]
    return $num_parts
} {3}

test tensor_split-2.3 {Named parameter syntax with dimension} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}}]
    set result [torch::tensor_split -input $tensor -sections 2 -dim 0]
    set num_parts [llength $result]
    return $num_parts
} {2}

test tensor_split-2.4 {Named parameter syntax with alternative names} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_split -tensor $tensor -sections 3 -dimension 0]
    set num_parts [llength $result]
    return $num_parts
} {3}

;# Test cases for camelCase alias
test tensor_split-3.1 {CamelCase alias - positional syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensorSplit $tensor 3]
    set num_parts [llength $result]
    return $num_parts
} {3}

test tensor_split-3.2 {CamelCase alias - named parameter syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensorSplit -input $tensor -sections 3]
    set num_parts [llength $result]
    return $num_parts
} {3}

;# Error handling tests
test tensor_split-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_split invalid_tensor 3} result
    return $result
} {Invalid tensor name}

test tensor_split-4.2 {Error handling - missing sections} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_split $tensor} result
    return $result
} {Error in tensor_split: Invalid number of arguments}

test tensor_split-4.3 {Error handling - empty sections} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_split $tensor {}} result
    return $result
} {Error in tensor_split: Required parameters missing: input tensor and sections/indices are required}

test tensor_split-4.4 {Error handling - invalid named parameter} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_split -invalid $tensor -sections 3} result
    return $result
} {Error in tensor_split: Unknown parameter: -invalid}

test tensor_split-4.5 {Error handling - missing parameter value} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_split -input $tensor -sections} result
    return $result
} {Error in tensor_split: Missing value for parameter}

test tensor_split-4.6 {Error handling - invalid dimension} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_split -input $tensor -sections 3 -dim invalid} result
    return $result
} {Error in tensor_split: Invalid dimension value}

;# Edge cases
test tensor_split-5.1 {Edge case - split into 1 part} {
    set tensor [torch::tensor_create {1 2 3}]
    set result [torch::tensor_split $tensor 1]
    set num_parts [llength $result]
    return $num_parts
} {1}

test tensor_split-5.2 {Edge case - split more parts than elements} {
    set tensor [torch::tensor_create {1 2 3}]
    set result [torch::tensor_split $tensor 5]
    set num_parts [llength $result]
    return $num_parts
} {5}

test tensor_split-5.3 {Edge case - split at boundaries} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result [torch::tensor_split $tensor {3 6}]
    set num_parts [llength $result]
    return $num_parts
} {3}

;# Data type tests
test tensor_split-6.1 {Data type - float tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
    set result [torch::tensor_split $tensor 2]
    set num_parts [llength $result]
    return $num_parts
} {2}

test tensor_split-6.2 {Data type - int tensor} {
    set tensor [torch::tensor_create {1 2 3 4 5 6} int64]
    set result [torch::tensor_split $tensor 3]
    set num_parts [llength $result]
    return $num_parts
} {3}

;# Multi-dimensional tensor tests
test tensor_split-7.1 {2D tensor - split along first dimension} {
    set tensor [torch::tensor_create {{1 2} {3 4} {5 6} {7 8}}]
    set result [torch::tensor_split $tensor 2 0]
    set num_parts [llength $result]
    return $num_parts
} {2}

test tensor_split-7.2 {2D tensor - split along second dimension} {
    set tensor [torch::tensor_create {{1 2 3 4} {5 6 7 8}}]
    set result [torch::tensor_split $tensor 2 1]
    set num_parts [llength $result]
    return $num_parts
} {2}

test tensor_split-7.3 {2D tensor - split by indices} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}}]
    set result [torch::tensor_split $tensor {1 3} 0]
    set num_parts [llength $result]
    return $num_parts
} {3}

;# Consistency tests - both syntaxes should produce same results
test tensor_split-8.1 {Consistency - positional vs named syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result1 [torch::tensor_split $tensor 3]
    set result2 [torch::tensor_split -input $tensor -sections 3]
    set num_parts1 [llength $result1]
    set num_parts2 [llength $result2]
    return [expr {$num_parts1 == $num_parts2}]
} {1}

test tensor_split-8.2 {Consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create {1 2 3 4 5 6}]
    set result1 [torch::tensor_split $tensor 3]
    set result2 [torch::tensorSplit $tensor 3]
    set num_parts1 [llength $result1]
    set num_parts2 [llength $result2]
    return [expr {$num_parts1 == $num_parts2}]
} {1}

;# Complex scenarios
test tensor_split-9.1 {Complex - 3D tensor split} {
    set tensor [torch::tensor_create {{1 2 3 4} {5 6 7 8} {9 10 11 12} {13 14 15 16}}]
    set result [torch::tensor_split $tensor 2 0]
    set num_parts [llength $result]
    return $num_parts
} {2}

test tensor_split-9.2 {Complex - split with multiple indices} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8 9 10}]
    set result [torch::tensor_split $tensor {2 5 8}]
    set num_parts [llength $result]
    return $num_parts
} {4}

cleanupTests 