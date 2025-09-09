#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Create test layer for testing
set layer_name [torch::linear 10 5]

;# Test 1: Basic positional syntax (backward compatibility)
test layer_device-1.1 {Basic positional syntax} {
    set result [torch::layer_device $layer_name]
    expr {$result ne ""}
} {1}

;# Test 2: Named parameter syntax
test layer_device-2.1 {Named parameter syntax with -layer} {
    set result [torch::layer_device -layer $layer_name]
    expr {$result ne ""}
} {1}

;# Test 3: Named parameter syntax with -input alias
test layer_device-2.2 {Named parameter syntax with -input alias} {
    set result [torch::layer_device -input $layer_name]
    expr {$result ne ""}
} {1}

;# Test 4: camelCase alias with positional syntax
test layer_device-3.1 {camelCase alias with positional syntax} {
    set result [torch::layerDevice $layer_name]
    expr {$result ne ""}
} {1}

;# Test 5: camelCase alias with named parameter syntax
test layer_device-3.2 {camelCase alias with named parameter syntax} {
    set result [torch::layerDevice -layer $layer_name]
    expr {$result ne ""}
} {1}

;# Test 6: Error handling - invalid layer name (positional)
test layer_device-4.1 {Error handling - invalid layer name (positional)} {
    catch {torch::layer_device "invalid_layer"} msg
    expr {[string match "*Invalid layer name*" $msg]}
} {1}

;# Test 7: Error handling - invalid layer name (named)
test layer_device-4.2 {Error handling - invalid layer name (named)} {
    catch {torch::layer_device -layer "invalid_layer"} msg
    expr {[string match "*Invalid layer name*" $msg]}
} {1}

;# Test 8: Error handling - missing parameter value
test layer_device-4.3 {Error handling - missing parameter value} {
    catch {torch::layer_device -layer} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} {1}

;# Test 9: Error handling - unknown parameter
test layer_device-4.4 {Error handling - unknown parameter} {
    catch {torch::layer_device -unknown_param value} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

;# Test 10: Error handling - missing required parameter
test layer_device-4.5 {Error handling - missing required parameter} {
    catch {torch::layer_device -layer ""} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

;# Test 11: Verify device detection works - CPU by default
test layer_device-5.1 {Verify device detection - CPU by default} {
    ;# Layer should start on CPU
    set device [torch::layer_device $layer_name]
    expr {$device eq "cpu"}
} {1}

;# Test 12: Device detection after moving to CPU
test layer_device-5.2 {Device detection after moving to CPU} {
    ;# Explicitly move to CPU
    torch::layer_cpu $layer_name
    ;# Check device
    set device [torch::layer_device $layer_name]
    expr {$device eq "cpu"}
} {1}

;# Test 13: Device detection after moving to CUDA (if available)
test layer_device-5.3 {Device detection after moving to CUDA} {
    if {![torch::cuda_is_available]} {
        skip "CUDA not available"
    }
    ;# Move to CUDA
    torch::layer_cuda $layer_name
    ;# Check device
    set device [torch::layer_device $layer_name]
    expr {[string match "cuda*" $device]}
} {1}

;# Test 14: Multiple parameter formats work identically
test layer_device-6.1 {Multiple parameter formats produce same result} {
    ;# Test that both syntaxes produce the same result
    set result1 [torch::layer_device $layer_name]
    set result2 [torch::layer_device -layer $layer_name]
    set result3 [torch::layerDevice $layer_name]
    set result4 [torch::layerDevice -layer $layer_name]
    expr {$result1 eq $result2 && $result2 eq $result3 && $result3 eq $result4}
} {1}

;# Test 15: Device transitions are properly detected
test layer_device-6.2 {Device transitions are properly detected} {
    ;# Start on CPU
    torch::layer_cpu $layer_name
    set cpu_device [torch::layer_device $layer_name]
    
    if {[torch::cuda_is_available]} {
        ;# Move to CUDA
        torch::layer_cuda $layer_name
        set cuda_device [torch::layer_device $layer_name]
        
        ;# Move back to CPU
        torch::layer_cpu $layer_name
        set cpu_device2 [torch::layer_device $layer_name]
        
        expr {$cpu_device eq "cpu" && [string match "cuda*" $cuda_device] && $cpu_device2 eq "cpu"}
    } else {
        ;# CUDA not available, just verify CPU
        expr {$cpu_device eq "cpu"}
    }
} {1}

;# Test 16: Return value format validation
test layer_device-6.3 {Return value format validation} {
    set device [torch::layer_device $layer_name]
    ;# Should be either "cpu" or "cuda:N" format
    expr {$device eq "cpu" || [string match "cuda:*" $device]}
} {1}

;# Test 17: Consistency across multiple layers
test layer_device-6.4 {Consistency across multiple layers} {
    ;# Create another layer
    set layer2 [torch::linear 5 3]
    
    ;# Both should start on CPU
    set device1 [torch::layer_device $layer_name]
    set device2 [torch::layer_device $layer2]
    
    expr {$device1 eq "cpu" && $device2 eq "cpu"}
} {1}

cleanupTests 