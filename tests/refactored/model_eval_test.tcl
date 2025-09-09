#!/usr/bin/env tclsh

# Test file for torch::model_eval command with dual syntax support
# Tests both positional and named parameter syntax

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create a simple model for testing
proc create_test_model {} {
    # Create a simple linear model
    set model [torch::sequential]
    
    # Add a mock model to the module_storage for testing
    # This is a simplified approach since we can't easily create a real model in TCL tests
    # The actual model_eval command only needs a valid model name in the module_storage
    
    return $model
}

# Test suite for torch::model_eval
test model_eval-1.1 {Basic positional syntax} {
    # Create a test model
    set model [create_test_model]
    
    # Set model to training mode first
    torch::model_train $model
    
    # Test basic model_eval with positional syntax
    set result [torch::model_eval $model]
    
    # Verify result is the model name
    expr {$result eq $model}
} {1}

test model_eval-2.1 {Named parameter syntax} {
    # Create a test model
    set model [create_test_model]
    
    # Set model to training mode first
    torch::model_train $model
    
    # Test model_eval with named parameter syntax
    set result [torch::model_eval -model $model]
    
    # Verify result is the model name
    expr {$result eq $model}
} {1}

test model_eval-3.1 {CamelCase alias} {
    # Create a test model
    set model [create_test_model]
    
    # Set model to training mode first
    torch::model_train $model
    
    # Test camelCase alias
    set result [torch::modelEval -model $model]
    
    # Verify result is the model name
    expr {$result eq $model}
} {1}

test model_eval-4.1 {Error handling - invalid model} {
    # Test with invalid model name
    catch {torch::model_eval "invalid_model"} err
    
    # Verify error message
    string match "*Invalid model name*" $err
} {1}

test model_eval-4.2 {Error handling - missing model parameter} {
    # Test with missing model parameter
    catch {torch::model_eval -model} err
    
    # Verify error message
    string match "*Missing value for parameter*" $err
} {1}

test model_eval-4.3 {Error handling - invalid parameter} {
    # Create a test model
    set model [create_test_model]
    
    # Test with invalid parameter
    catch {torch::model_eval -invalid $model} err
    
    # Verify error message
    string match "*Unknown parameter*" $err
} {1}

cleanupTests
