# tests/refactored/autocast_disable_test.tcl
# Test file for refactored torch::autocast_disable command

# Explicitly load the built shared library for local testing
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Warning: Could not load ../../build/libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::autocast_disable]]} {
    puts "❌ Command torch::autocast_disable not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "autocast_disable"
set COMMAND_OLD "torch::autocast_disable"
set COMMAND_NEW "torch::autocastDisable"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax (backward compatibility) - no arguments (default)
puts "Test 1: Original positional syntax - no arguments..."
if {[catch {
    set result1 [$COMMAND_OLD]
    puts "  Original syntax (default): OK - $result1"
} result]} {
    puts "  ❌ Original syntax (default) failed: $result"
    exit 1
}

# Test 2: Original positional syntax with device type
puts "Test 2: Original positional syntax with device type..."
if {[catch {
    set result2 [$COMMAND_OLD cuda]
    puts "  Original syntax (cuda): OK - $result2"
} result]} {
    puts "  ❌ Original syntax (cuda) failed: $result"
    exit 1
}

# Test 3: Original positional syntax with CPU device type
puts "Test 3: Original positional syntax with CPU device type..."
if {[catch {
    set result3 [$COMMAND_OLD cpu]
    puts "  Original syntax (cpu): OK - $result3"
} result]} {
    puts "  ❌ Original syntax (cpu) failed: $result"
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

# Test 5: New named parameter syntax with -device
puts "Test 5: New named parameter syntax with -device..."
if {[catch {
    set result5 [$COMMAND_OLD -device cpu]
    puts "  Named parameters (-device cpu): OK - $result5"
} result]} {
    puts "  ❌ Named parameters (-device) failed: $result"
    exit 1
}

# Test 6: camelCase alias with positional syntax
puts "Test 6: camelCase alias with positional syntax..."
if {[catch {
    set result6 [$COMMAND_NEW cuda]
    puts "  camelCase alias (positional): OK - $result6"
} result]} {
    puts "  ❌ camelCase alias (positional) failed: $result"
    exit 1
}

# Test 7: camelCase alias with named parameters
puts "Test 7: camelCase alias with named parameters..."
if {[catch {
    set result7 [$COMMAND_NEW -device_type cpu]
    puts "  camelCase alias (named): OK - $result7"
} result]} {
    puts "  ❌ camelCase alias (named) failed: $result"
    exit 1
}

# Test 8: Check autocast status for CUDA
puts "Test 8: Check autocast status for CUDA..."
if {[catch {
    set status_cuda [torch::autocast_is_enabled cuda]
    $COMMAND_OLD cuda
    puts "  Autocast status check for CUDA: OK (current status: $status_cuda)"
} result]} {
    puts "  ❌ CUDA autocast status check failed: $result"
    exit 1
}

# Test 9: Check autocast status for CPU
puts "Test 9: Check autocast status for CPU..."
if {[catch {
    set status_cpu [torch::autocast_is_enabled cpu]
    $COMMAND_OLD cpu
    puts "  Autocast status check for CPU: OK (current status: $status_cpu)"
} result]} {
    puts "  ❌ CPU autocast status check failed: $result"
    exit 1
}

# Test 10: Error handling - invalid device type
puts "Test 10: Error handling - invalid device type..."
if {[catch {
    $COMMAND_OLD invalid_device
    puts "  ❌ Should have failed with invalid device type"
    exit 1
} result]} {
    puts "  Error handling (positional): OK - $result"
}

# Test 11: Error handling - invalid device type with named parameters
puts "Test 11: Error handling - invalid device type with named parameters..."
if {[catch {
    $COMMAND_OLD -device_type invalid_device
    puts "  ❌ Should have failed with invalid device type"
    exit 1
} result]} {
    puts "  Error handling (named): OK - $result"
}

# Test 12: Error handling - invalid parameter names
puts "Test 12: Error handling - invalid parameter names..."
if {[catch {
    $COMMAND_OLD -invalid_param cuda
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling (invalid param): OK - $result"
}

# Test 13: Error handling - missing value for parameter
puts "Test 13: Error handling - missing value for parameter..."
if {[catch {
    $COMMAND_OLD -device_type
    puts "  ❌ Should have failed with missing value"
    exit 1
} result]} {
    puts "  Error handling (missing value): OK - $result"
}

# Test 14: Consistency between syntaxes
puts "Test 14: Consistency between syntaxes..."
if {[catch {
    # Test different syntaxes produce same result (without enabling first)
    set result_pos [$COMMAND_OLD cuda]
    set result_named [$COMMAND_OLD -device_type cuda]
    set result_camel [$COMMAND_NEW -device cuda]
    
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

# Test 15: Performance comparison
puts "Test 15: Performance comparison..."
set iterations 1000
puts "  Running $iterations iterations..."

if {[catch {
    # Test old syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD cuda
    }
    set end [clock clicks -milliseconds]
    set old_time [expr {$end - $start}]
    
    # Test new syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD -device_type cuda
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

# Test 16: Both device types work
puts "Test 16: Both device types work correctly..."
if {[catch {
    # Test CUDA device
    set result_cuda [$COMMAND_OLD -device_type cuda]
    
    # Test CPU device
    set result_cpu [$COMMAND_OLD -device cpu]
    
    if {$result_cuda eq "autocast disabled" && $result_cpu eq "autocast disabled"} {
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