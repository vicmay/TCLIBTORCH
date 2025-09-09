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
test layer_cpu-1.1 {Basic positional syntax} {
    set result [torch::layer_cpu $layer_name]
    expr {$result eq $layer_name}
} {1}

;# Test 2: Named parameter syntax
test layer_cpu-2.1 {Named parameter syntax with -layer} {
    set result [torch::layer_cpu -layer $layer_name]
    expr {$result eq $layer_name}
} {1}

;# Test 3: Named parameter syntax with -input alias
test layer_cpu-2.2 {Named parameter syntax with -input alias} {
    set result [torch::layer_cpu -input $layer_name]
    expr {$result eq $layer_name}
} {1}

;# Test 4: camelCase alias with positional syntax
test layer_cpu-3.1 {camelCase alias with positional syntax} {
    set result [torch::layerCpu $layer_name]
    expr {$result eq $layer_name}
} {1}

;# Test 5: camelCase alias with named parameter syntax
test layer_cpu-3.2 {camelCase alias with named parameter syntax} {
    set result [torch::layerCpu -layer $layer_name]
    expr {$result eq $layer_name}
} {1}

;# Test 6: Error handling - invalid layer name (positional)
test layer_cpu-4.1 {Error handling - invalid layer name (positional)} {
    catch {torch::layer_cpu "invalid_layer"} msg
    expr {[string match "*Invalid layer name*" $msg]}
} {1}

;# Test 7: Error handling - invalid layer name (named)
test layer_cpu-4.2 {Error handling - invalid layer name (named)} {
    catch {torch::layer_cpu -layer "invalid_layer"} msg
    expr {[string match "*Invalid layer name*" $msg]}
} {1}

;# Test 8: Error handling - missing parameter value
test layer_cpu-4.3 {Error handling - missing parameter value} {
    catch {torch::layer_cpu -layer} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} {1}

;# Test 9: Error handling - unknown parameter
test layer_cpu-4.4 {Error handling - unknown parameter} {
    catch {torch::layer_cpu -unknown_param value} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

;# Test 10: Error handling - missing required parameter
test layer_cpu-4.5 {Error handling - missing required parameter} {
    catch {torch::layer_cpu -layer ""} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

;# Test 11: Verify device movement works
test layer_cpu-5.1 {Verify device movement to CPU} {
    ;# First move to CPU (should work)
    torch::layer_cpu $layer_name
    ;# Check device - should be CPU
    set device [torch::layer_device $layer_name]
    expr {$device eq "cpu"}
} {1}

;# Test 12: Chaining operations (layer_cpu returns layer name)
test layer_cpu-5.2 {Chaining operations} {
    set result [torch::layer_cpu $layer_name]
    ;# Should return the layer name for chaining
    expr {$result eq $layer_name}
} {1}

;# Test 13: Multiple parameter formats work identically
test layer_cpu-6.1 {Multiple parameter formats produce same result} {
    ;# Test that both syntaxes produce the same result
    set result1 [torch::layer_cpu $layer_name]
    set result2 [torch::layer_cpu -layer $layer_name]
    set result3 [torch::layerCpu $layer_name]
    set result4 [torch::layerCpu -layer $layer_name]
    expr {$result1 eq $result2 && $result2 eq $result3 && $result3 eq $result4}
} {1}

cleanupTests 