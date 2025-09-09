#!/usr/bin/env tclsh

# Test file for torch::load_state command with dual syntax support
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
proc setup_test_module {} {
    # Create a simple linear module for testing
    set module [torch::linear 10 5]
    return $module
}

proc setup_test_file {module_name} {
    # Create a temporary file for testing
    set temp_file "/tmp/test_model_${module_name}.pt"
    
    # Save the module state first (needed for load testing)
    if {[catch {torch::save_state $module_name $temp_file}]} {
        return ""
    }
    
    return $temp_file
}

proc cleanup_test_file {filename} {
    if {[file exists $filename]} {
        file delete $filename
    }
}

# Test suite for torch::load_state
test load_state-1.1 {Basic positional syntax} {
    # Setup test module and file
    set module_name [setup_test_module]
    set temp_file [setup_test_file $module_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test basic load_state with positional syntax
    set result [catch {torch::load_state $module_name $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state-2.1 {Named parameter syntax - basic} {
    # Setup test module and file
    set module_name [setup_test_module]
    set temp_file [setup_test_file $module_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test basic named parameter syntax
    set result [catch {torch::load_state -module $module_name -filename $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state-2.2 {Named parameter syntax with -file alias} {
    # Setup test module and file
    set module_name [setup_test_module]
    set temp_file [setup_test_file $module_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test named parameter syntax with -file alias
    set result [catch {torch::load_state -module $module_name -file $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state-3.1 {CamelCase alias - basic} {
    # Setup test module and file
    set module_name [setup_test_module]
    set temp_file [setup_test_file $module_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test camelCase alias
    set result [catch {torch::loadState -module $module_name -filename $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state-3.2 {CamelCase alias with -file parameter} {
    # Setup test module and file
    set module_name [setup_test_module]
    set temp_file [setup_test_file $module_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test camelCase alias with -file parameter
    set result [catch {torch::loadState -module $module_name -file $temp_file} error]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Should succeed (return 0 for no error)
    expr {$result == 0}
} {1}

test load_state-4.1 {Error handling - invalid module name} {
    # Create a test file
    set temp_file "/tmp/test_invalid_module.pt"
    
    # Test with non-existent module
    set result [catch {torch::load_state invalid_module $temp_file} error]
    
    # Should fail (return 1 for error) and error message should contain useful info
    expr {$result == 1 && [string length $error] > 0}
} {1}

test load_state-4.2 {Error handling - missing filename} {
    # Setup test module
    set module_name [setup_test_module]
    
    # Test with missing filename parameter
    set result [catch {torch::load_state -module $module_name} error]
    
    # Should fail and error message should mention missing parameter
    expr {$result == 1 && [string match "*missing*" [string tolower $error]]}
} {1}

test load_state-4.3 {Error handling - missing module parameter} {
    # Test with missing module parameter
    set result [catch {torch::load_state -filename "/tmp/test.pt"} error]
    
    # Should fail and error message should mention missing parameter
    expr {$result == 1 && [string match "*missing*" [string tolower $error]]}
} {1}

test load_state-4.4 {Error handling - unknown parameter} {
    # Setup test module
    set module_name [setup_test_module]
    
    # Test with unknown parameter
    set result [catch {torch::load_state -module $module_name -filename "/tmp/test.pt" -unknown_param value} error]
    
    # Should fail and error message should mention unknown parameter
    expr {$result == 1 && [string match "*unknown*parameter*" [string tolower $error]]}
} {1}

test load_state-4.5 {Error handling - insufficient positional args} {
    # Test with only one positional argument
    set result [catch {torch::load_state single_arg} error]
    
    # Should fail
    expr {$result == 1}
} {1}

test load_state-4.6 {Error handling - too many positional args} {
    # Test with too many positional arguments
    set result [catch {torch::load_state module file extra_arg} error]
    
    # Should fail
    expr {$result == 1}
} {1}

test load_state-5.1 {Parameter validation - both syntaxes equivalent} {
    # Setup test module and file
    set module_name [setup_test_module]
    set temp_file [setup_test_file $module_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test both syntaxes should behave the same
    set result1 [catch {torch::load_state $module_name $temp_file} error1]
    set result2 [catch {torch::load_state -module $module_name -filename $temp_file} error2]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Both should succeed or both should fail with same result
    expr {$result1 == $result2}
} {1}

test load_state-5.2 {Parameter validation - camelCase vs snake_case} {
    # Setup test module and file
    set module_name [setup_test_module]
    set temp_file [setup_test_file $module_name]
    
    if {$temp_file eq ""} {
        return 0  ;# Skip test if setup failed
    }
    
    # Test snake_case and camelCase should behave the same
    set result1 [catch {torch::load_state -module $module_name -filename $temp_file} error1]
    set result2 [catch {torch::loadState -module $module_name -filename $temp_file} error2]
    
    # Cleanup
    cleanup_test_file $temp_file
    
    # Both should succeed or both should fail with same result
    expr {$result1 == $result2}
} {1}

test load_state-6.1 {File handling - nonexistent file} {
    # Setup test module
    set module_name [setup_test_module]
    
    # Test with non-existent file
    set result [catch {torch::load_state $module_name "/tmp/nonexistent_file.pt"} error]
    
    # Should fail
    expr {$result == 1}
} {1}

test load_state-6.2 {File handling - empty filename} {
    # Setup test module
    set module_name [setup_test_module]
    
    # Test with empty filename
    set result [catch {torch::load_state -module $module_name -filename ""} error]
    
    # Should fail and error should mention required parameters
    expr {$result == 1 && [string match "*required*" [string tolower $error]]}
} {1}

test load_state-6.3 {File handling - empty module name} {
    # Test with empty module name
    set result [catch {torch::load_state -module "" -filename "/tmp/test.pt"} error]
    
    # Should fail and error should mention required parameters
    expr {$result == 1 && [string match "*required*" [string tolower $error]]}
} {1}

# Cleanup tests
cleanupTests 