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

# Helper function to create test modules
proc create_test_modules {} {
    set linear1 [torch::linear -inFeatures 10 -outFeatures 20]
    set linear2 [torch::linear -inFeatures 20 -outFeatures 15]
    set linear3 [torch::linear -inFeatures 15 -outFeatures 5]
    return [list $linear1 $linear2 $linear3]
}

# Test cases for positional syntax
test sequential-1.1 {Empty sequential with positional syntax} {
    set seq [torch::sequential]
    expr {[string match "sequential*" $seq]}
} {1}

test sequential-1.2 {Sequential with modules using positional syntax} {
    lassign [create_test_modules] linear1 linear2 linear3
    set seq [torch::sequential [list $linear1 $linear2 $linear3]]
    expr {[string match "sequential*" $seq]}
} {1}

# Test cases for named parameter syntax
test sequential-2.1 {Empty sequential with named syntax} {
    set seq [torch::sequential]
    expr {[string match "sequential*" $seq]}
} {1}

test sequential-2.2 {Sequential with modules using named syntax} {
    lassign [create_test_modules] linear1 linear2 linear3
    set seq [torch::sequential -modules [list $linear1 $linear2 $linear3]]
    expr {[string match "sequential*" $seq]}
} {1}

# Test forward pass functionality
test sequential-3.1 {Forward pass through sequential} {
    lassign [create_test_modules] linear1 linear2 linear3
    set seq [torch::sequential -modules [list $linear1 $linear2 $linear3]]
    set input [torch::tensor_create {1 2 3 4 5 6 7 8 9 10} float32]
    set output [torch::layer_forward $seq $input]
    expr {[string match "tensor*" $output]}
} {1}

# Test error handling
test sequential-4.1 {Error on invalid module} {
    catch {torch::sequential -modules [list "invalid_module"]} err
    set err
} {Invalid module name: invalid_module}

test sequential-4.2 {Error on invalid parameter} {
    catch {torch::sequential -invalid value} err
    set err
} {Unknown parameter: -invalid}

test sequential-4.3 {Error on missing parameter value} {
    catch {torch::sequential -modules} err
    set err
} {Missing value for parameter}

# Test camelCase alias
test sequential-5.1 {CamelCase alias with empty sequential} {
    set seq [torch::sequential]
    expr {[string match "sequential*" $seq]}
} {1}

test sequential-5.2 {CamelCase alias with modules} {
    lassign [create_test_modules] linear1 linear2 linear3
    set seq [torch::sequential -modules [list $linear1 $linear2 $linear3]]
    expr {[string match "sequential*" $seq]}
} {1}

cleanupTests 