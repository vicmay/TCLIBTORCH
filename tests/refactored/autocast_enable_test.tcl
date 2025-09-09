# tests/refactored/autocast_enable_test.tcl
# Test file for refactored torch::autocast_enable command

# Explicitly load the built shared library for local testing
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Warning: Could not load ../../build/libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::autocast_enable]]} {
    puts "❌ Command torch::autocast_enable not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "autocast_enable"
set COMMAND_OLD "torch::autocast_enable"
set COMMAND_NEW "torch::autocastEnable"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax - no arguments (defaults)
puts "Test 1: Original positional syntax - no arguments..."
if {[catch {
    set result1 [$COMMAND_OLD]
    puts "  Original syntax (defaults): OK - $result1"
} result]} {
    puts "  ❌ Original syntax (defaults) failed: $result"
    exit 1
}

# Test 2: Original positional syntax with device type only
puts "Test 2: Original positional syntax with device type..."
if {[catch {
    set result2 [$COMMAND_OLD cuda]
    puts "  Original syntax (cuda): OK - $result2"
} result]} {
    puts "  ❌ Original syntax (cuda) failed: $result"
    exit 1
}

# Test 3: Original positional syntax with both parameters
puts "Test 3: Original positional syntax with both parameters..."
if {[catch {
    set result3 [$COMMAND_OLD cpu float32]
    puts "  Original syntax (cpu float32): OK - $result3"
} result]} {
    puts "  ❌ Original syntax (cpu float32) failed: $result"
    exit 1
}

# Test 4: New named parameter syntax with -device_type
puts "Test 4: New named parameter syntax with -device_type..."
if {[catch {
    set result4 [$COMMAND_OLD -device_type cuda]
    puts "  Named parameters (-device_type cuda): OK - $result4"
} result]} {
    puts "  ❌ Named parameters (-device_type) failed: $result"
    exit 1
}

# Test 5: New named parameter syntax with -device and -dtype
puts "Test 5: New named parameter syntax with -device and -dtype..."
if {[catch {
    set result5 [$COMMAND_OLD -device cpu -dtype bfloat16]
    puts "  Named parameters (-device cpu -dtype bfloat16): OK - $result5"
} result]} {
    puts "  ❌ Named parameters (-device -dtype) failed: $result"
    exit 1
}

# Test 6: New named parameter syntax with -data_type
puts "Test 6: New named parameter syntax with -data_type..."
if {[catch {
    set result6 [$COMMAND_OLD -device_type cuda -data_type float16]
    puts "  Named parameters (-data_type): OK - $result6"
} result]} {
    puts "  ❌ Named parameters (-data_type) failed: $result"
    exit 1
}

# Test 7: camelCase alias with positional syntax
puts "Test 7: camelCase alias with positional syntax..."
if {[catch {
    set result7 [$COMMAND_NEW cuda float16]
    puts "  camelCase alias (positional): OK - $result7"
} result]} {
    puts "  ❌ camelCase alias (positional) failed: $result"
    exit 1
}

# Test 8: camelCase alias with named parameters
puts "Test 8: camelCase alias with named parameters..."
if {[catch {
    set result8 [$COMMAND_NEW -device_type cpu -dtype float32]
    puts "  camelCase alias (named): OK - $result8"
} result]} {
    puts "  ❌ camelCase alias (named) failed: $result"
    exit 1
}

# Test 9: Check that autocast is enabled for CUDA
puts "Test 9: Check autocast status for CUDA..."
if {[catch {
    $COMMAND_OLD -device cuda -dtype float16
    set status_cuda [torch::autocast_is_enabled cuda]
    puts "  Autocast status check for CUDA: OK (status: $status_cuda)"
} result]} {
    puts "  ❌ CUDA autocast status check failed: $result"
    exit 1
}

# Test 10: Check that autocast is enabled for CPU
puts "Test 10: Check autocast status for CPU..."
if {[catch {
    $COMMAND_OLD -device cpu -dtype bfloat16
    set status_cpu [torch::autocast_is_enabled cpu]
    puts "  Autocast status check for CPU: OK (status: $status_cpu)"
} result]} {
    puts "  ❌ CPU autocast status check failed: $result"
    exit 1
}

# Test 11: Different data types
puts "Test 11: Different data types..."
if {[catch {
    # Test float16
    set result_f16 [$COMMAND_OLD -device cuda -dtype float16]
    
    # Test bfloat16
    set result_bf16 [$COMMAND_OLD -device cuda -dtype bfloat16]
    
    # Test float32
    set result_f32 [$COMMAND_OLD -device cuda -dtype float32]
    
    puts "  Different data types: OK (f16: $result_f16, bf16: $result_bf16, f32: $result_f32)"
} result]} {
    puts "  ❌ Different data types test failed: $result"
    exit 1
}

# Test 12: Parameter order independence
puts "Test 12: Parameter order independence..."
if {[catch {
    # Different parameter orders should work
    set result1 [$COMMAND_OLD -device cuda -dtype float16]
    set result2 [$COMMAND_OLD -dtype float16 -device cuda]
    
    if {$result1 eq $result2} {
        puts "  Parameter order independence: OK"
    } else {
        puts "  ❌ Parameter order affects results"
        exit 1
    }
} result]} {
    puts "  ❌ Parameter order test failed: $result"
    exit 1
}

# Test 13: Error handling - invalid device type
puts "Test 13: Error handling - invalid device type..."
if {[catch {
    $COMMAND_OLD invalid_device
    puts "  ❌ Should have failed with invalid device type"
    exit 1
} result]} {
    puts "  Error handling (invalid device): OK - $result"
}

# Test 14: Error handling - invalid dtype
puts "Test 14: Error handling - invalid dtype..."
if {[catch {
    $COMMAND_OLD -device cuda -dtype invalid_dtype
    puts "  ❌ Should have failed with invalid dtype"
    exit 1
} result]} {
    puts "  Error handling (invalid dtype): OK - $result"
}

# Test 15: Error handling - invalid parameter names
puts "Test 15: Error handling - invalid parameter names..."
if {[catch {
    $COMMAND_OLD -invalid_param cuda
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling (invalid param): OK - $result"
}

# Test 16: Error handling - missing value for parameter
puts "Test 16: Error handling - missing value for parameter..."
if {[catch {
    $COMMAND_OLD -device_type
    puts "  ❌ Should have failed with missing value"
    exit 1
} result]} {
    puts "  Error handling (missing value): OK - $result"
}

# Test 17: Error handling - too many positional arguments
puts "Test 17: Error handling - too many positional arguments..."
if {[catch {
    $COMMAND_OLD cuda float16 extra_param
    puts "  ❌ Should have failed with too many arguments"
    exit 1
} result]} {
    puts "  Error handling (too many args): OK - $result"
}

# Test 18: Consistency between syntaxes
puts "Test 18: Consistency between syntaxes..."
if {[catch {
    # Test different syntaxes produce same result
    set result_pos [$COMMAND_OLD cuda float16]
    set result_named [$COMMAND_OLD -device_type cuda -dtype float16]
    set result_camel [$COMMAND_NEW -device cuda -data_type float16]
    
    if {$result_pos eq $result_named && $result_named eq $result_camel} {
        puts "  Consistency between syntaxes: OK (all return: $result_pos)"
    } else {
        puts "  ❌ Results differ between syntaxes"
        exit 1
    }
} result]} {
    puts "  ❌ Consistency test failed: $result"
    exit 1
}

# Test 19: Performance comparison
puts "Test 19: Performance comparison..."
set iterations 1000
puts "  Running $iterations iterations..."

if {[catch {
    # Test old syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD cuda float16
    }
    set end [clock clicks -milliseconds]
    set old_time [expr {$end - $start}]
    
    # Test new syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD -device_type cuda -dtype float16
    }
    set end [clock clicks -milliseconds]
    set new_time [expr {$end - $start}]
    
    puts "  Old syntax: ${old_time}ms"
    puts "  New syntax: ${new_time}ms"
    
    # Check that performance is reasonable (less than 5x difference)
    if {$new_time <= [expr {$old_time * 5}] && $old_time <= [expr {$new_time * 5}]} {
        puts "  Performance: OK (within acceptable range)"
    } else {
        puts "  ⚠️  Performance difference may be too large"
    }
} result]} {
    puts "  ❌ Performance test failed: $result"
    exit 1
}

# Test 20: Both device types work correctly
puts "Test 20: Both device types work correctly..."
if {[catch {
    # Test CUDA device
    set result_cuda [$COMMAND_OLD -device_type cuda -dtype float16]
    
    # Test CPU device
    set result_cpu [$COMMAND_OLD -device cpu -dtype bfloat16]
    
    if {$result_cuda eq "autocast enabled" && $result_cpu eq "autocast enabled"} {
        puts "  Both device types work: OK"
    } else {
        puts "  ❌ Device types don't work properly: cuda=$result_cuda, cpu=$result_cpu"
        exit 1
    }
} result]} {
    puts "  ❌ Device types test failed: $result"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Update tracking: mark command complete in database"
puts "   3. Commit changes: git add ."
puts "   4. Move to next command: python3 scripts/query_next_commands.py" 