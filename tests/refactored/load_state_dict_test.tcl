#!/usr/bin/env tclsh

# Test file for torch::load_state_dict command with dual syntax support
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

# Setup test environment
proc setup_test_model {} {
    # Create a simple linear module for testing
    set model [torch::linear 10 5]
    return $model
}

proc setup_test_file {model_name} {
    # Create a temporary file for testing
    set temp_file "/tmp/test_state_dict_${model_name}.pt"
    
    # Save the model state dict first (needed for load testing)
    if {[catch {torch::save_state_dict $model_name $temp_file}]} {
        return ""
    }
    
    return $temp_file
}

proc cleanup_test_file {filename} {
    if {[file exists $filename]} {
        file delete $filename
    }
}

# Test suite for torch::load_state_dict
test load_state_dict-1.1 {Basic positional syntax} {
    # Setup test model and file
    set model_name [setup_test_model]
    set temp_file [setup_test_file $model_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test basic load_state_dict with positional syntax
    set result [catch {torch::load_state_dict $model_name $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state_dict-2.1 {Named parameter syntax - basic} {
    # Setup test model and file
    set model_name [setup_test_model]
    set temp_file [setup_test_file $model_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test basic named parameter syntax
    set result [catch {torch::load_state_dict -model $model_name -filename $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state_dict-2.2 {Named parameter syntax with -file alias} {
    # Setup test model and file
    set model_name [setup_test_model]
    set temp_file [setup_test_file $model_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test named parameter syntax with -file alias
    set result [catch {torch::load_state_dict -model $model_name -file $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state_dict-3.1 {CamelCase alias - basic} {
    # Setup test model and file
    set model_name [setup_test_model]
    set temp_file [setup_test_file $model_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test camelCase alias
    set result [catch {torch::loadStateDict -model $model_name -filename $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state_dict-3.2 {CamelCase alias with -file parameter} {
    # Setup test model and file
    set model_name [setup_test_model]
    set temp_file [setup_test_file $model_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test camelCase alias with -file parameter
    set result [catch {torch::loadStateDict -model $model_name -file $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state_dict-4.1 {Error handling - invalid model name} {
    # Create a test file
    set temp_file "/tmp/test_invalid_state_dict.pt"
    
    # Test with non-existent model
    set result [catch {torch::load_state_dict invalid_model $temp_file} error]
    
    # Should fail (return 1 for error) and error message should contain useful info
    expr {$result == 1 && [string length $error] > 0}
} {1}

test load_state_dict-4.2 {Error handling - missing filename} {
    # Setup test model
    set model_name [setup_test_model]
    
    # Test with missing filename parameter
    set result [catch {torch::load_state_dict -model $model_name} error]
    
    # Should fail and error message should mention missing parameter
    expr {$result == 1 && [string match "*missing*" [string tolower $error]]}
} {1}

test load_state_dict-4.3 {Error handling - missing model parameter} {
    # Test with missing model parameter
    set result [catch {torch::load_state_dict -filename "/tmp/test.pt"} error]
    
    # Should fail and error message should mention missing parameter
    expr {$result == 1 && [string match "*missing*" [string tolower $error]]}
} {1}

test load_state_dict-4.4 {Error handling - unknown parameter} {
    # Setup test model
    set model_name [setup_test_model]
    
    # Test with unknown parameter
    set result [catch {torch::load_state_dict -model $model_name -filename "/tmp/test.pt" -unknown_param value} error]
    
    # Should fail and error message should mention unknown parameter
    expr {$result == 1 && [string match "*unknown*parameter*" [string tolower $error]]}
} {1}

test load_state_dict-4.5 {Error handling - insufficient positional args} {
    # Test with only one positional argument
    set result [catch {torch::load_state_dict single_arg} error]
    
    # Should fail
    expr {$result == 1}
} {1}

test load_state_dict-4.6 {Error handling - too many positional args} {
    # Test with too many positional arguments
    set result [catch {torch::load_state_dict model file extra_arg} error]
    
    # Should fail
    expr {$result == 1}
} {1}

test load_state_dict-5.1 {Parameter validation - both syntaxes equivalent} {
    # Setup test model and file
    set model_name [setup_test_model]
    set temp_file [setup_test_file $model_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test both syntaxes should behave the same
    set result1 [catch {torch::load_state_dict $model_name $temp_file} error1]
    set result2 [catch {torch::load_state_dict -model $model_name -filename $temp_file} error2]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Both should succeed or both should fail with same result
    expr {$result1 == $result2}
} {1}

test load_state_dict-5.2 {Parameter validation - camelCase vs snake_case} {
    # Setup test model and file
    set model_name [setup_test_model]
    set temp_file [setup_test_file $model_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test snake_case and camelCase should behave the same
    set result1 [catch {torch::load_state_dict -model $model_name -filename $temp_file} error1]
    set result2 [catch {torch::loadStateDict -model $model_name -filename $temp_file} error2]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Both should succeed or both should fail with same result
    expr {$result1 == $result2}
} {1}

test load_state_dict-6.1 {File handling - nonexistent file} {
    # Setup test model
    set model_name [setup_test_model]
    
    # Test with non-existent file
    set result [catch {torch::load_state_dict $model_name "/tmp/nonexistent_state_dict.pt"} error]
    
    # Should fail
    expr {$result == 1}
} {1}

test load_state_dict-6.2 {File handling - empty filename} {
    # Setup test model
    set model_name [setup_test_model]
    
    # Test with empty filename
    set result [catch {torch::load_state_dict -model $model_name -filename ""} error]
    
    # Should fail and error should mention required parameters
    expr {$result == 1 && [string match "*required*" [string tolower $error]]}
} {1}

test load_state_dict-6.3 {File handling - empty model name} {
    # Test with empty model name
    set result [catch {torch::load_state_dict -model "" -filename "/tmp/test.pt"} error]
    
    # Should fail and error should mention required parameters
    expr {$result == 1 && [string match "*required*" [string tolower $error]]}
} {1}

test load_state_dict-7.1 {State dict vs regular state - functional difference} {
    # Create two identical models
    set model1 [torch::linear 10 5]
    set model2 [torch::linear 10 5]
    
    # Save using state dict method
    set state_dict_file "/tmp/test_state_dict.pt"
    set state_file "/tmp/test_state.pt"
    
    # Save both ways
    set result1 [catch {torch::save_state_dict $model1 $state_dict_file} error1]
    set result2 [catch {torch::save_state $model1 $state_file} error2]
    
    # Both should succeed
    set both_saves_ok [expr {$result1 == 0 && $result2 == 0}]
    
    # Test loading state dict
    set result3 [catch {torch::load_state_dict $model2 $state_dict_file} error3]
    
    # Cleanup
    cleanup_test_file $state_dict_file
    cleanup_test_file $state_file
    
    # State dict load should succeed
    expr {$both_saves_ok && $result3 == 0}
} {1}

test load_state_dict-7.2 {Multiple models - independent state dicts} {
    # Create different models
    set model1 [torch::linear 10 5]
    set model2 [torch::linear 20 10]
    
    # Save their state dicts
    set file1 "/tmp/model1_state_dict.pt"
    set file2 "/tmp/model2_state_dict.pt"
    
    set result1 [catch {torch::save_state_dict $model1 $file1} error1]
    set result2 [catch {torch::save_state_dict $model2 $file2} error2]
    
    # Load them back
    set result3 [catch {torch::load_state_dict $model1 $file1} error3]
    set result4 [catch {torch::load_state_dict $model2 $file2} error4]
    
    # Cleanup
    cleanup_test_file $file1
    cleanup_test_file $file2
    
    # All operations should succeed
    expr {$result1 == 0 && $result2 == 0 && $result3 == 0 && $result4 == 0}
} {1}

# Cleanup tests
cleanupTests 