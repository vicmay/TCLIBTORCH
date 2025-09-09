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
test save_state_dict-1.1 {Basic positional syntax} {
    set filename "test_state_dict_1.pt"
    set result [torch::save_state_dict $model $filename]
    file exists $filename
} {1}

test save_state_dict-1.2 {Positional syntax - verify result} {
    set filename "test_state_dict_2.pt"
    set result [torch::save_state_dict $model $filename]
    string match "Model state dict saved to: *" $result
} {1}

# Test cases for named parameter syntax
test save_state_dict-2.1 {Named parameter syntax - with -model} {
    set filename "test_state_dict_3.pt"
    set result [torch::save_state_dict -model $model -filename $filename]
    file exists $filename
} {1}

test save_state_dict-2.2 {Named parameter syntax - with -file} {
    set filename "test_state_dict_4.pt"
    set result [torch::save_state_dict -model $model -file $filename]
    file exists $filename
} {1}

test save_state_dict-2.3 {Named parameter syntax - verify result} {
    set filename "test_state_dict_5.pt"
    set result [torch::save_state_dict -model $model -filename $filename]
    string match "Model state dict saved to: *" $result
} {1}

# Test cases for camelCase alias
test save_state_dict-3.1 {CamelCase alias - with -model} {
    set filename "test_state_dict_6.pt"
    set result [torch::saveStateDict -model $model -filename $filename]
    file exists $filename
} {1}

test save_state_dict-3.2 {CamelCase alias - with -file} {
    set filename "test_state_dict_7.pt"
    set result [torch::saveStateDict -model $model -file $filename]
    file exists $filename
} {1}

# Error handling tests
test save_state_dict-4.1 {Error - no arguments} {
    catch {torch::save_state_dict} err
    set err
} {wrong # args: should be "torch::save_state_dict model filename"}

test save_state_dict-4.2 {Error - too few arguments} {
    catch {torch::save_state_dict $model} err
    set err
} {wrong # args: should be "torch::save_state_dict model filename"}

test save_state_dict-4.3 {Error - too many arguments} {
    catch {torch::save_state_dict $model file1 extra} err
    set err
} {wrong # args: should be "torch::save_state_dict model filename"}

test save_state_dict-4.4 {Error - invalid model} {
    catch {torch::save_state_dict invalid_model "test.pt"} err
    set err
} {Model not found}

test save_state_dict-4.5 {Error - missing value for parameter} {
    catch {torch::save_state_dict -model} err
    set err
} {Error in save_state_dict: Missing value for parameter}

test save_state_dict-4.6 {Error - unknown parameter} {
    catch {torch::save_state_dict -invalid $model} err
    set err
} {Error in save_state_dict: Unknown parameter: -invalid}

test save_state_dict-4.7 {Error - missing required parameter} {
    catch {torch::save_state_dict -file "test.pt"} err
    set err
} {Error in save_state_dict: Required parameters missing: -model and -filename}

# Cleanup test files
foreach file [glob -nocomplain test_state_dict_*.pt] {
    file delete $file
}

cleanupTests 