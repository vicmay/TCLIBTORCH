#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test output
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic functionality - positional syntax
test exponential-1.1 {Basic exponential positional syntax} {
    set result [torch::exponential {2 3} 1.0]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-1.2 {Basic exponential named parameter syntax} {
    set result [torch::exponential -size {2 3} -rate 1.0]
    expr {[string match "tensor*" $result]}
} {1}

# Test 2: Different sizes
test exponential-2.1 {Single dimension} {
    set result [torch::exponential {5} 2.0]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-2.2 {Multi-dimensional with named parameters} {
    set result [torch::exponential -size {3 4} -rate 1.5]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-2.3 {Three-dimensional tensor} {
    set result [torch::exponential -size {2 3 4} -rate 0.5]
    expr {[string match "tensor*" $result]}
} {1}

# Test 3: Rate parameter effects
test exponential-3.1 {Different rates produce valid tensors} {
    set result1 [torch::exponential {100} 0.1]
    set result2 [torch::exponential {100} 10.0]
    
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test exponential-3.2 {Rate parameter in named syntax} {
    set result [torch::exponential -size {5} -rate 2.0]
    expr {[string match "tensor*" $result]}
} {1}

# Test 4: Data type support
test exponential-4.1 {Float32 data type - positional} {
    set result [torch::exponential {3} 1.0 float32]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-4.2 {Float64 data type - named} {
    set result [torch::exponential -size {3} -rate 1.0 -dtype float64]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-4.3 {Default data type} {
    set result [torch::exponential -size {2} -rate 1.0]
    expr {[string match "tensor*" $result]}
} {1}

# Test 5: Device support
test exponential-5.1 {CPU device - positional} {
    set result [torch::exponential {2} 1.0 float32 cpu]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-5.2 {CPU device - named} {
    set result [torch::exponential -size {2} -rate 1.0 -device cpu]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-5.3 {Default device} {
    set result [torch::exponential -size {2} -rate 1.0]
    expr {[string match "tensor*" $result]}
} {1}

# Test 6: Statistical properties (basic validation)
test exponential-6.1 {All values should be positive} {
    set result [torch::exponential {100} 1.0]
    set min_val [torch::tensor_min $result]
    set min_value [torch::tensor_item $min_val]
    expr {$min_value > 0.0}
} {1}

test exponential-6.2 {Mean should be approximately 1/rate for rate=1} {
    set result [torch::exponential {10000} 1.0]
    set mean_tensor [torch::tensor_mean $result]
    set mean_value [torch::tensor_item $mean_tensor]
    # For exponential distribution, mean = 1/rate = 1/1 = 1
    # Allow tolerance for random variation
    expr {abs($mean_value - 1.0) < 0.1}
} {1}

test exponential-6.3 {Mean should be approximately 1/rate for rate=2} {
    set result [torch::exponential {10000} 2.0]
    set mean_tensor [torch::tensor_mean $result]
    set mean_value [torch::tensor_item $mean_tensor]
    # For exponential distribution, mean = 1/rate = 1/2 = 0.5
    expr {abs($mean_value - 0.5) < 0.05}
} {1}

# Test 7: Error handling
test exponential-7.1 {Error: Missing arguments} {
    catch {torch::exponential} result
    expr {[string match "*Usage*" $result] || [string match "*Required parameters*" $result]}
} {1}

test exponential-7.2 {Error: Invalid rate (negative)} {
    catch {torch::exponential {2} -1.0} result
    expr {[string match "*positive*" $result] || [string match "*rate*" $result]}
} {1}

test exponential-7.3 {Error: Invalid rate (zero)} {
    catch {torch::exponential {2} 0.0} result
    expr {[string match "*positive*" $result] || [string match "*rate*" $result]}
} {1}

test exponential-7.4 {Error: Named parameter without value} {
    catch {torch::exponential -size} result
    expr {[string match "*Usage*" $result]}
} {1}

test exponential-7.5 {Error: Unknown parameter} {
    catch {torch::exponential -size {2} -rate 1.0 -unknown value} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test exponential-7.6 {Error: Missing required parameter size} {
    catch {torch::exponential -rate 1.0} result
    expr {[string match "*Required parameters*" $result]}
} {1}

test exponential-7.7 {Default rate parameter when not specified} {
    set result [torch::exponential -size {2}]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-7.8 {Error: Invalid dtype} {
    catch {torch::exponential -size {2} -rate 1.0 -dtype invalid_type} result
    expr {[string match "*Invalid dtype*" $result] || [string match "*error*" $result]}
} {1}

test exponential-7.9 {Fallback to default device when invalid device specified} {
    set result [torch::exponential -size {2} -rate 1.0 -device invalid_device]
    expr {[string match "tensor*" $result]}
} {1}

# Test 8: Syntax consistency
test exponential-8.1 {Both syntaxes produce valid tensors} {
    set result_pos [torch::exponential {3 4} 1.0]
    set result_named [torch::exponential -size {3 4} -rate 1.0]
    
    expr {[string match "tensor*" $result_pos] && [string match "tensor*" $result_named]}
} {1}

test exponential-8.2 {Both syntaxes with data types} {
    set result_pos [torch::exponential {2} 1.0 float64]
    set result_named [torch::exponential -size {2} -rate 1.0 -dtype float64]
    
    expr {[string match "tensor*" $result_pos] && [string match "tensor*" $result_named]}
} {1}

# Test 9: Different rates
test exponential-9.1 {Very small rate} {
    set result [torch::exponential {5} 0.01]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-9.2 {Very large rate} {
    set result [torch::exponential -size {5} -rate 100.0]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-9.3 {Fractional rate} {
    set result [torch::exponential -size {3} -rate 1.5]
    expr {[string match "tensor*" $result]}
} {1}

# Test 10: Edge cases
test exponential-10.1 {Single element tensor} {
    set result [torch::exponential {1} 1.0]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-10.2 {Large tensor} {
    set result [torch::exponential -size {100 100} -rate 1.0]
    expr {[string match "tensor*" $result]}
} {1}

test exponential-10.3 {Positional syntax with all parameters} {
    set result [torch::exponential {2} 1.0 float32 cpu]
    expr {[string match "tensor*" $result]}
} {1}

cleanupTests 