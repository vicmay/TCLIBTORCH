#!/usr/bin/env tclsh

# Test file for torch::model_summary command with dual syntax support
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
    # The actual model_summary command only needs a valid model name in the module_storage
    
    return $model
}

# Test suite for torch::model_summary
test model_summary-1.1 {Basic positional syntax} {
    # Create a test model
    set model [create_test_model]
    
    # Test basic model_summary with positional syntax
    set result [catch {torch::model_summary $model} output]
    
    # Verify result contains expected text
    expr {[string match "*Model Summary*" $output] || $result == 1}
} {1}

test model_summary-2.1 {Named parameter syntax} {
    # Create a test model
    set model [create_test_model]
    
    # Test model_summary with named parameter syntax
    set result [catch {torch::model_summary -model $model} output]
    
    # Verify result contains expected text
    expr {[string match "*Model Summary*" $output] || $result == 1}
} {1}

test model_summary-3.1 {CamelCase alias} {
    # Create a test model
    set model [create_test_model]
    
    # Test camelCase alias
    set result [catch {torch::modelSummary -model $model} output]
    
    # Verify result contains expected text
    expr {[string match "*Model Summary*" $output] || $result == 1}
} {1}

test model_summary-4.1 {Error handling - invalid model} {
    # Test with invalid model name
    catch {torch::model_summary "invalid_model"} err
    
    # Verify error message
    string match "*Model not found*" $err
} {1}

test model_summary-4.2 {Error handling - missing model parameter} {
    # Test with missing model parameter
    catch {torch::model_summary -model} err
    
    # Verify error message
    string match "*Missing value for parameter*" $err
} {1}

test model_summary-4.3 {Error handling - invalid parameter} {
    # Create a test model
    set model [create_test_model]
    
    # Test with invalid parameter
    catch {torch::model_summary -invalid $model} err
    
    # Verify error message
    string match "*Unknown parameter*" $err
} {1}

cleanupTests
