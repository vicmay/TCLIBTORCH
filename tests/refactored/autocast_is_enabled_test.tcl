#!/usr/bin/env tclsh
# tests/refactored/autocast_is_enabled_test.tcl
# Test file for refactored torch::autocast_is_enabled command

# Explicitly load the built shared library for local testing
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Warning: Could not load ../../build/libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::autocast_is_enabled]]} {
    puts "❌ Command torch::autocast_is_enabled not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "autocast_is_enabled"
set COMMAND_OLD "torch::autocast_is_enabled"
set COMMAND_NEW "torch::autocastIsEnabled"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax - no arguments (defaults to CUDA)
puts "Test 1: Original positional syntax - no arguments..."
if {[catch {
    set result1 [$COMMAND_OLD]
    puts "  Original syntax (defaults): OK - $result1"
} result]} {
    puts "  ❌ Original syntax (defaults) failed: $result"
    exit 1
}

# Test 2: Original positional syntax with CUDA device
puts "Test 2: Original positional syntax with CUDA device..."
if {[catch {
    set result2 [$COMMAND_OLD cuda]
    puts "  Original syntax (cuda): OK - $result2"
} result]} {
    puts "  ❌ Original syntax (cuda) failed: $result"
    exit 1
}

# Test 3: Original positional syntax with CPU device
puts "Test 3: Original positional syntax with CPU device..."
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

# Test 8: Integration with autocast_enable for CUDA
puts "Test 8: Integration with autocast_enable for CUDA..."
if {[catch {
    # First disable autocast to ensure clean state
    torch::autocast_disable cuda
    set disabled_status [$COMMAND_OLD cuda]
    
    # Enable autocast for CUDA
    torch::autocast_enable cuda float16
    set enabled_status [$COMMAND_OLD cuda]
    
    if {$disabled_status == 0 && $enabled_status == 1} {
        puts "  Integration test CUDA: OK (disabled=$disabled_status, enabled=$enabled_status)"
    } else {
        puts "  ❌ Integration test CUDA failed: disabled=$disabled_status, enabled=$enabled_status"
        exit 1
    }
} result]} {
    puts "  ❌ Integration test CUDA failed: $result"
    exit 1
}

# Test 9: Integration with autocast_enable for CPU
puts "Test 9: Integration with autocast_enable for CPU..."
if {[catch {
    # First disable autocast to ensure clean state
    torch::autocast_disable cpu
    set disabled_status [$COMMAND_OLD cpu]
    
    # Enable autocast for CPU
    torch::autocast_enable cpu bfloat16
    set enabled_status [$COMMAND_OLD cpu]
    
    if {$disabled_status == 0 && $enabled_status == 1} {
        puts "  Integration test CPU: OK (disabled=$disabled_status, enabled=$enabled_status)"
    } else {
        puts "  ❌ Integration test CPU failed: disabled=$disabled_status, enabled=$enabled_status"
        exit 1
    }
} result]} {
    puts "  ❌ Integration test CPU failed: $result"
    exit 1
}

# Test 10: Return type verification
puts "Test 10: Return type verification..."
if {[catch {
    set result_true [$COMMAND_OLD cuda]
    set result_false [$COMMAND_OLD cuda]
    
    # Check that we get boolean values (0 or 1)
    if {($result_true == 0 || $result_true == 1) && ($result_false == 0 || $result_false == 1)} {
        puts "  Return type verification: OK (returns boolean values)"
    } else {
        puts "  ❌ Return type verification failed: not boolean values"
        exit 1
    }
} result]} {
    puts "  ❌ Return type verification failed: $result"
    exit 1
}

# Test 11: Consistency between syntaxes
puts "Test 11: Consistency between syntaxes..."
if {[catch {
    # Enable autocast first
    torch::autocast_enable cuda float16
    
    # Test different syntaxes produce same result
    set result_pos [$COMMAND_OLD cuda]
    set result_named [$COMMAND_OLD -device_type cuda]
    set result_camel [$COMMAND_NEW -device cuda]
    
    if {$result_pos == $result_named && $result_named == $result_camel} {
        puts "  Consistency between syntaxes: OK (all return: $result_pos)"
    } else {
        puts "  ❌ Results differ between syntaxes: pos=$result_pos, named=$result_named, camel=$result_camel"
        exit 1
    }
} result]} {
    puts "  ❌ Consistency test failed: $result"
    exit 1
}

# Test 12: Error handling - invalid device type
puts "Test 12: Error handling - invalid device type..."
if {[catch {
    $COMMAND_OLD invalid_device
    puts "  ❌ Should have failed with invalid device type"
    exit 1
} result]} {
    puts "  Error handling (invalid device): OK - $result"
}

# Test 13: Error handling - invalid parameter names
puts "Test 13: Error handling - invalid parameter names..."
if {[catch {
    $COMMAND_OLD -invalid_param cuda
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling (invalid param): OK - $result"
}

# Test 14: Error handling - missing value for parameter
puts "Test 14: Error handling - missing value for parameter..."
if {[catch {
    $COMMAND_OLD -device_type
    puts "  ❌ Should have failed with missing value"
    exit 1
} result]} {
    puts "  Error handling (missing value): OK - $result"
}

# Test 15: Error handling - too many positional arguments
puts "Test 15: Error handling - too many positional arguments..."
if {[catch {
    $COMMAND_OLD cuda extra_param
    puts "  ❌ Should have failed with too many arguments"
    exit 1
} result]} {
    puts "  Error handling (too many args): OK - $result"
}

# Test 16: Both devices work independently
puts "Test 16: Both devices work independently..."
if {[catch {
    # Enable CUDA autocast, disable CPU autocast
    torch::autocast_enable cuda float16
    torch::autocast_disable cpu
    
    set cuda_status [$COMMAND_OLD cuda]
    set cpu_status [$COMMAND_OLD cpu]
    
    if {$cuda_status == 1 && $cpu_status == 0} {
        puts "  Independent device states: OK (cuda=$cuda_status, cpu=$cpu_status)"
    } else {
        puts "  ❌ Device states are not independent: cuda=$cuda_status, cpu=$cpu_status"
        exit 1
    }
} result]} {
    puts "  ❌ Independent device test failed: $result"
    exit 1
}

# Test 17: Performance comparison
puts "Test 17: Performance comparison..."
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

# Test 18: State persistence across calls
puts "Test 18: State persistence across calls..."
if {[catch {
    # Enable autocast
    torch::autocast_enable cuda float16
    
    # Check multiple times - should remain enabled
    set status1 [$COMMAND_OLD cuda]
    set status2 [$COMMAND_OLD -device cuda]
    set status3 [$COMMAND_NEW cuda]
    
    if {$status1 == 1 && $status2 == 1 && $status3 == 1} {
        puts "  State persistence: OK (all calls return enabled state)"
    } else {
        puts "  ❌ State not persistent: $status1, $status2, $status3"
        exit 1
    }
} result]} {
    puts "  ❌ State persistence test failed: $result"
    exit 1
}

# Test 19: Edge case - rapid enable/disable/check cycles
puts "Test 19: Edge case - rapid enable/disable/check cycles..."
if {[catch {
    set all_correct 1
    
    for {set i 0} {$i < 10} {incr i} {
        torch::autocast_enable cuda float16
        set enabled_status [$COMMAND_OLD cuda]
        
        torch::autocast_disable cuda
        set disabled_status [$COMMAND_OLD cuda]
        
        if {$enabled_status != 1 || $disabled_status != 0} {
            set all_correct 0
            break
        }
    }
    
    if {$all_correct} {
        puts "  Rapid cycles: OK (all enable/disable cycles work correctly)"
    } else {
        puts "  ❌ Rapid cycles failed"
        exit 1
    }
} result]} {
    puts "  ❌ Rapid cycles test failed: $result"
    exit 1
}

# Test 20: Comprehensive workflow test
puts "Test 20: Comprehensive workflow test..."
if {[catch {
    # Test complete autocast workflow
    
    # 1. Start with autocast disabled
    torch::autocast_disable cuda
    torch::autocast_disable cpu
    
    set initial_cuda [$COMMAND_OLD cuda]
    set initial_cpu [$COMMAND_OLD cpu]
    
    # 2. Enable autocast for both devices
    torch::autocast_enable cuda float16
    torch::autocast_enable cpu bfloat16
    
    set enabled_cuda [$COMMAND_OLD -device_type cuda]
    set enabled_cpu [$COMMAND_NEW -device cpu]
    
    # 3. Disable one device
    torch::autocast_disable cuda
    
    set final_cuda [$COMMAND_OLD cuda]
    set final_cpu [$COMMAND_OLD cpu]
    
    # Verify workflow
    if {$initial_cuda == 0 && $initial_cpu == 0 && 
        $enabled_cuda == 1 && $enabled_cpu == 1 && 
        $final_cuda == 0 && $final_cpu == 1} {
        puts "  Comprehensive workflow: OK"
        puts "    Initial: cuda=$initial_cuda, cpu=$initial_cpu"
        puts "    Enabled: cuda=$enabled_cuda, cpu=$enabled_cpu"
        puts "    Final: cuda=$final_cuda, cpu=$final_cpu"
    } else {
        puts "  ❌ Comprehensive workflow failed"
        exit 1
    }
} result]} {
    puts "  ❌ Comprehensive workflow test failed: $result"
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