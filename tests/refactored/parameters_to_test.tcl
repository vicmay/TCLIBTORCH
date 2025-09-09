#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create test parameters
proc create_test_parameters {} {
    set param1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set param2 [torch::tensor_create {4.0 5.0 6.0} float32]
    return [list $param1 $param2]
}

# Test cases for positional syntax
test parameters-1.1 {Basic positional syntax - move to CPU} {
    set params [create_test_parameters]
    torch::parameters_to $params cpu
    set result "ok"
} {ok}

test parameters-1.2 {Basic positional syntax - default device} {
    set params [create_test_parameters]
    torch::parameters_to $params
    set result "ok"
} {ok}

# Test cases for named syntax
test parameters-2.1 {Named parameter syntax - move to CPU} {
    set params [create_test_parameters]
    torch::parameters_to -parameters $params -device cpu
    set result "ok"
} {ok}

test parameters-2.2 {Named parameter syntax - default device} {
    set params [create_test_parameters]
    torch::parameters_to -parameters $params
    set result "ok"
} {ok}

# Test cases for camelCase alias
test parameters-3.1 {CamelCase alias - positional syntax} {
    set params [create_test_parameters]
    torch::parametersTo $params cpu
    set result "ok"
} {ok}

test parameters-3.2 {CamelCase alias - named syntax} {
    set params [create_test_parameters]
    torch::parametersTo -parameters $params -device cpu
    set result "ok"
} {ok}

# Error handling tests
test parameters-4.1 {Missing required parameters} {
    catch {torch::parameters_to} result
    set result
} {Error in parameters_to: Required parameters missing or invalid (parameters required, device must be 'cpu' or 'cuda')}

test parameters-4.2 {Invalid parameters list} {
    catch {torch::parameters_to invalid_params} result
    set result
} {Invalid parameter tensor: invalid_params}

test parameters-4.3 {Invalid device} {
    set params [create_test_parameters]
    catch {torch::parameters_to $params invalid_device} result
    set result
} {Error in parameters_to: Required parameters missing or invalid (parameters required, device must be 'cpu' or 'cuda')}

cleanupTests 