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

# Create test model
set model [torch::linear 10 5]

# Test cases for positional syntax
test save_state-1.1 {Basic positional syntax} {
    set filename "test_state_1.pt"
    set result [torch::save_state $model $filename]
    file exists $filename
} {1}

test save_state-1.2 {Positional syntax - verify result} {
    set filename "test_state_2.pt"
    set result [torch::save_state $model $filename]
    set result
} {OK}

# Test cases for named parameter syntax
test save_state-2.1 {Named parameter syntax - with -module} {
    set filename "test_state_3.pt"
    set result [torch::save_state -module $model -filename $filename]
    file exists $filename
} {1}

test save_state-2.2 {Named parameter syntax - with -file} {
    set filename "test_state_4.pt"
    set result [torch::save_state -module $model -file $filename]
    file exists $filename
} {1}

test save_state-2.3 {Named parameter syntax - verify result} {
    set filename "test_state_5.pt"
    set result [torch::save_state -module $model -filename $filename]
    set result
} {OK}

# Test cases for camelCase alias
test save_state-3.1 {CamelCase alias - with -module} {
    set filename "test_state_6.pt"
    set result [torch::saveState -module $model -filename $filename]
    file exists $filename
} {1}

test save_state-3.2 {CamelCase alias - with -file} {
    set filename "test_state_7.pt"
    set result [torch::saveState -module $model -file $filename]
    file exists $filename
} {1}

# Error handling tests
test save_state-4.1 {Error - no arguments} {
    catch {torch::save_state} err
    set err
} {wrong # args: should be "torch::save_state module filename"}

test save_state-4.2 {Error - too few arguments} {
    catch {torch::save_state $model} err
    set err
} {wrong # args: should be "torch::save_state module filename"}

test save_state-4.3 {Error - too many arguments} {
    catch {torch::save_state $model file1 extra} err
    set err
} {wrong # args: should be "torch::save_state module filename"}

test save_state-4.4 {Error - invalid module} {
    catch {torch::save_state invalid_model "test.pt"} err
    set err
} {Invalid module name}

test save_state-4.5 {Error - missing value for parameter} {
    catch {torch::save_state -module} err
    set err
} {Error in save_state: Missing value for parameter}

test save_state-4.6 {Error - unknown parameter} {
    catch {torch::save_state -invalid $model} err
    set err
} {Error in save_state: Unknown parameter: -invalid}

test save_state-4.7 {Error - missing required parameter} {
    catch {torch::save_state -file "test.pt"} err
    set err
} {Error in save_state: Required parameters missing: -module and -filename}

# Cleanup test files
foreach file [glob -nocomplain test_state_*.pt] {
    file delete $file
}

cleanupTests 