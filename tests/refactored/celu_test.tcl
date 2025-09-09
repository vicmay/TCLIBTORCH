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
test celu-1.1 {Basic positional syntax with default alpha} {
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result [torch::celu $t1]
    expr {$result ne ""}
} {1}

test celu-1.2 {Positional syntax with custom alpha} {
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result [torch::celu $t1 0.5]
    expr {$result ne ""}
} {1}

# Test cases for named parameter syntax
test celu-2.1 {Named parameters with default alpha} {
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result [torch::celu -input $t1]
    expr {$result ne ""}
} {1}

test celu-2.2 {Named parameters with custom alpha} {
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result [torch::celu -input $t1 -alpha 0.5]
    expr {$result ne ""}
} {1}

test celu-2.3 {Alternative parameter names} {
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result [torch::celu -tensor $t1 -alpha 1.5]
    expr {$result ne ""}
} {1}

# Error handling tests
test celu-3.1 {Error on missing input} {
    catch {torch::celu} result
    expr {[string match "*parameter*missing*" $result] || [string match "*Usage*" $result]}
} {1}

test celu-3.2 {Error on invalid alpha (negative)} {
    set t1 [torch::tensor_create {1.0 2.0} float32]
    catch {torch::celu $t1 -0.5} result
    expr {[string match "*alpha*" $result] || [string match "*parameter*" $result]}
} {1}

test celu-3.3 {Error on unknown parameter} {
    set t1 [torch::tensor_create {1.0 2.0} float32]
    catch {torch::celu -input $t1 -unknown_param 1.0} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

cleanupTests 