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

# Helper function to compare tensor values
proc compare_tensor_values {values expected} {
    if {[llength $values] != [llength $expected]} {
        return 0
    }
    foreach v $values e $expected {
        if {abs($v - $e) > 1e-6} {
            return 0
        }
    }
    return 1
}

# Create test tensor
set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3} -dtype float32]

# Test cases for positional syntax
test rot90-1.1 {Basic positional syntax - default k=1} {
    set result [torch::rot90 $tensor]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {3 6 2 5 1 4}
} {1}

test rot90-1.2 {Positional syntax with k=2} {
    set result [torch::rot90 $tensor 2]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {6 5 4 3 2 1}
} {1}

test rot90-1.3 {Positional syntax with k=1 and dims} {
    set result [torch::rot90 $tensor 1 {0 1}]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {3 6 2 5 1 4}
} {1}

# Test cases for named parameter syntax
test rot90-2.1 {Named parameter syntax - default k} {
    set result [torch::rot90 -input $tensor]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {3 6 2 5 1 4}
} {1}

test rot90-2.2 {Named parameter syntax with k} {
    set result [torch::rot90 -input $tensor -k 2]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {6 5 4 3 2 1}
} {1}

test rot90-2.3 {Named parameter syntax with k and dims} {
    set result [torch::rot90 -input $tensor -k 1 -dims {0 1}]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {3 6 2 5 1 4}
} {1}

# Test cases for camelCase alias
test rot90-3.1 {CamelCase alias - basic usage} {
    set result [torch::Rot90 -input $tensor]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {3 6 2 5 1 4}
} {1}

# Error handling tests
test rot90-4.1 {Error - invalid tensor} {
    catch {torch::rot90 invalid_tensor} err
    set err
} {Invalid input tensor}

test rot90-4.2 {Error - invalid k value} {
    catch {torch::rot90 -input $tensor -k invalid} err
    set err
} {Invalid k value}

test rot90-4.3 {Error - invalid dims list} {
    catch {torch::rot90 -input $tensor -dims invalid} err
    set err
} {Invalid dims list}

cleanupTests 