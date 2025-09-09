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
test vsplit-1.1 {Basic positional syntax with sections} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vsplit $tensor 2]
    set num_tensors [llength $result]
    return $num_tensors
} {2}

test vsplit-1.2 {Positional syntax with indices} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vsplit $tensor {1 3}]
    set num_tensors [llength $result]
    return $num_tensors
} {3}

test vsplit-1.3 {Positional syntax with 2D tensor} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vsplit $tensor 2]
    set num_tensors [llength $result]
    return $num_tensors
} {2}

;# Test cases for named parameter syntax
test vsplit-2.1 {Named parameter syntax with -tensor and -sections} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vsplit -tensor $tensor -sections 2]
    set num_tensors [llength $result]
    return $num_tensors
} {2}

test vsplit-2.2 {Named parameter syntax with -input and -indices} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vsplit -input $tensor -indices {2 4}]
    set num_tensors [llength $result]
    return $num_tensors
} {3}

test vsplit-2.3 {Named parameter syntax with 2D tensor} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}} float32 cpu false]
    set result [torch::vsplit -tensor $tensor -sections 3]
    set num_tensors [llength $result]
    return $num_tensors
} {3}

;# Test cases for camelCase alias
test vsplit-3.1 {CamelCase alias with sections} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vSplit $tensor 2]
    set num_tensors [llength $result]
    return $num_tensors
} {2}

test vsplit-3.2 {CamelCase alias with indices} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vSplit $tensor {1 3 5}]
    set num_tensors [llength $result]
    return $num_tensors
} {4}

test vsplit-3.3 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vSplit -tensor $tensor -sections 2]
    set num_tensors [llength $result]
    return $num_tensors
} {2}

;# Error handling tests
test vsplit-4.1 {Error handling - missing tensor} {
    catch {torch::vsplit} result
    return [string match "*Usage:*" $result]
} {1}

test vsplit-4.2 {Error handling - missing sections} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}} float32 cpu false]
    catch {torch::vsplit $tensor} result
    return [string match "*Usage:*" $result]
} {1}

test vsplit-4.3 {Error handling - invalid tensor handle} {
    catch {torch::vsplit invalid_handle 2} result
    return [string match "*Error in vsplit*" $result]
} {1}

test vsplit-4.4 {Error handling - unknown named parameter} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}} float32 cpu false]
    catch {torch::vsplit -invalid $tensor -sections 2} result
    return [string match "*Unknown parameter*" $result]
} {1}

;# Edge cases
test vsplit-5.1 {Edge case - split into 1 section} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vsplit $tensor 1]
    set num_tensors [llength $result]
    return $num_tensors
} {1}

test vsplit-5.2 {Edge case - split at boundaries} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result [torch::vsplit $tensor {0 4}]
    set num_tensors [llength $result]
    return $num_tensors
} {3}

;# Verify both syntaxes produce same results
test vsplit-6.1 {Consistency between positional and named syntax} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}} float32 cpu false]
    set result1 [torch::vsplit $tensor 2]
    set result2 [torch::vsplit -tensor $tensor -sections 2]
    set num1 [llength $result1]
    set num2 [llength $result2]
    return [expr {$num1 == $num2}]
} {1}

cleanupTests 