#!/usr/bin/env tclsh
# tests/refactored/autocast_set_dtype_test.tcl
# Test file for refactored torch::autocast_set_dtype command

# Explicitly load the built shared library for local testing
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Warning: Could not load ../../build/libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::autocast_set_dtype]]} {
    puts "❌ Command torch::autocast_set_dtype not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "autocast_set_dtype"
set COMMAND_OLD "torch::autocast_set_dtype"
set COMMAND_NEW "torch::autocastSetDtype"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax - dtype only (defaults to CUDA)
puts "Test 1: Original positional syntax - dtype only..."
if {[catch {
    set result1 [$COMMAND_OLD float16]
    puts "  Original syntax (float16): OK - $result1"
} result]} {
    puts "  ❌ Original syntax (float16) failed: $result"
    exit 1
}

# Test 2: Original positional syntax with both parameters - CUDA
puts "Test 2: Original positional syntax with both parameters - CUDA..."
if {[catch {
    set result2 [$COMMAND_OLD bfloat16 cuda]
    puts "  Original syntax (bfloat16 cuda): OK - $result2"
} result]} {
    puts "  ❌ Original syntax (bfloat16 cuda) failed: $result"
    exit 1
}

# Test 3: Original positional syntax with both parameters - CPU
puts "Test 3: Original positional syntax with both parameters - CPU..."
if {[catch {
    set result3 [$COMMAND_OLD float32 cpu]
    puts "  Original syntax (float32 cpu): OK - $result3"
} result]} {
    puts "  ❌ Original syntax (float32 cpu) failed: $result"
    exit 1
}

# Test 4: New named parameter syntax with -dtype only
puts "Test 4: New named parameter syntax with -dtype only..."
if {[catch {
    set result4 [$COMMAND_OLD -dtype float16]
    puts "  Named parameters (-dtype float16): OK - $result4"
} result]} {
    puts "  ❌ Named parameters (-dtype) failed: $result"
    exit 1
}

# Test 5: New named parameter syntax with -data_type
puts "Test 5: New named parameter syntax with -data_type..."
if {[catch {
    set result5 [$COMMAND_OLD -data_type bfloat16]
    puts "  Named parameters (-data_type bfloat16): OK - $result5"
} result]} {
    puts "  ❌ Named parameters (-data_type) failed: $result"
    exit 1
}

# Test 6: New named parameter syntax with both parameters
puts "Test 6: New named parameter syntax with both parameters..."
if {[catch {
    set result6 [$COMMAND_OLD -dtype float32 -device_type cpu]
    puts "  Named parameters (-dtype float32 -device_type cpu): OK - $result6"
} result]} {
    puts "  ❌ Named parameters (both) failed: $result"
    exit 1
}

# Test 7: New named parameter syntax with short parameter names
puts "Test 7: New named parameter syntax with short parameter names..."
if {[catch {
    set result7 [$COMMAND_OLD -data_type float16 -device cuda]
    puts "  Named parameters (short names): OK - $result7"
} result]} {
    puts "  ❌ Named parameters (short names) failed: $result"
    exit 1
}

# Test 8: camelCase alias with positional syntax
puts "Test 8: camelCase alias with positional syntax..."
if {[catch {
    set result8 [$COMMAND_NEW bfloat16 cuda]
    puts "  camelCase alias (positional): OK - $result8"
} result]} {
    puts "  ❌ camelCase alias (positional) failed: $result"
    exit 1
}

# Test 9: camelCase alias with named parameters
puts "Test 9: camelCase alias with named parameters..."
if {[catch {
    set result9 [$COMMAND_NEW -dtype float32 -device_type cpu]
    puts "  camelCase alias (named): OK - $result9"
} result]} {
    puts "  ❌ camelCase alias (named) failed: $result"
    exit 1
}

# Test 10: Parameter order independence
puts "Test 10: Parameter order independence..."
if {[catch {
    # Different parameter orders should work
    set result1 [$COMMAND_OLD -dtype float16 -device_type cuda]
    set result2 [$COMMAND_OLD -device_type cuda -dtype float16]
    
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

# Test 11: All supported data types
puts "Test 11: All supported data types..."
if {[catch {
    # Test float16
    set result_f16 [$COMMAND_OLD -dtype float16 -device cuda]
    
    # Test bfloat16
    set result_bf16 [$COMMAND_OLD -dtype bfloat16 -device cuda]
    
    # Test float32
    set result_f32 [$COMMAND_OLD -dtype float32 -device cuda]
    
    puts "  All data types: OK (f16: $result_f16, bf16: $result_bf16, f32: $result_f32)"
} result]} {
    puts "  ❌ Data types test failed: $result"
    exit 1
}

# Test 12: Both device types work
puts "Test 12: Both device types work..."
if {[catch {
    # Test CUDA device
    set result_cuda [$COMMAND_OLD -dtype float16 -device cuda]
    
    # Test CPU device
    set result_cpu [$COMMAND_OLD -dtype bfloat16 -device cpu]
    
    if {$result_cuda eq "autocast dtype set" && $result_cpu eq "autocast dtype set"} {
        puts "  Both device types: OK"
    } else {
        puts "  ❌ Device types don't work properly: cuda=$result_cuda, cpu=$result_cpu"
        exit 1
    }
} result]} {
    puts "  ❌ Device types test failed: $result"
    exit 1
}

# Test 13: Consistency between syntaxes
puts "Test 13: Consistency between syntaxes..."
if {[catch {
    # Test different syntaxes produce same result
    set result_pos [$COMMAND_OLD float16 cuda]
    set result_named [$COMMAND_OLD -dtype float16 -device_type cuda]
    set result_camel [$COMMAND_NEW -data_type float16 -device cuda]
    
    if {$result_pos eq $result_named && $result_named eq $result_camel} {
        puts "  Consistency between syntaxes: OK (all return: $result_pos)"
    } else {
        puts "  ❌ Results differ between syntaxes: pos=$result_pos, named=$result_named, camel=$result_camel"
        exit 1
    }
} result]} {
    puts "  ❌ Consistency test failed: $result"
    exit 1
}

# Test 14: Integration with autocast_enable and autocast_is_enabled
puts "Test 14: Integration with autocast commands..."
if {[catch {
    # Enable autocast first
    torch::autocast_enable cuda float16
    set enabled_before [torch::autocast_is_enabled cuda]
    
    # Change dtype using our command
    set set_result [$COMMAND_OLD -dtype bfloat16 -device cuda]
    
    # Verify autocast is still enabled
    set enabled_after [torch::autocast_is_enabled cuda]
    
    if {$enabled_before == 1 && $enabled_after == 1 && $set_result eq "autocast dtype set"} {
        puts "  Integration test: OK (before=$enabled_before, after=$enabled_after, set=$set_result)"
    } else {
        puts "  ❌ Integration test failed: before=$enabled_before, after=$enabled_after, set=$set_result"
        exit 1
    }
} result]} {
    puts "  ❌ Integration test failed: $result"
    exit 1
}

# Test 15: Error handling - missing dtype parameter
puts "Test 15: Error handling - missing dtype parameter..."
if {[catch {
    $COMMAND_OLD
    puts "  ❌ Should have failed with missing dtype"
    exit 1
} result]} {
    puts "  Error handling (missing dtype): OK - $result"
}

# Test 16: Error handling - invalid dtype
puts "Test 16: Error handling - invalid dtype..."
if {[catch {
    $COMMAND_OLD invalid_dtype
    puts "  ❌ Should have failed with invalid dtype"
    exit 1
} result]} {
    puts "  Error handling (invalid dtype): OK - $result"
}

# Test 17: Error handling - invalid device type
puts "Test 17: Error handling - invalid device type..."
if {[catch {
    $COMMAND_OLD float16 invalid_device
    puts "  ❌ Should have failed with invalid device type"
    exit 1
} result]} {
    puts "  Error handling (invalid device): OK - $result"
}

# Test 18: Error handling - invalid parameter names
puts "Test 18: Error handling - invalid parameter names..."
if {[catch {
    $COMMAND_OLD -invalid_param float16
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling (invalid param): OK - $result"
}

# Test 19: Error handling - missing value for parameter
puts "Test 19: Error handling - missing value for parameter..."
if {[catch {
    $COMMAND_OLD -dtype
    puts "  ❌ Should have failed with missing value"
    exit 1
} result]} {
    puts "  Error handling (missing value): OK - $result"
}

# Test 20: Error handling - too many positional arguments
puts "Test 20: Error handling - too many positional arguments..."
if {[catch {
    $COMMAND_OLD float16 cuda extra_param
    puts "  ❌ Should have failed with too many arguments"
    exit 1
} result]} {
    puts "  Error handling (too many args): OK - $result"
}

# Test 21: Error handling - missing dtype in named syntax
puts "Test 21: Error handling - missing dtype in named syntax..."
if {[catch {
    $COMMAND_OLD -device_type cuda
    puts "  ❌ Should have failed with missing required dtype"
    exit 1
} result]} {
    puts "  Error handling (missing required dtype): OK - $result"
}

# Test 22: Performance comparison
puts "Test 22: Performance comparison..."
set iterations 1000
puts "  Running $iterations iterations..."

if {[catch {
    # Test old syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD float16 cuda
    }
    set end [clock clicks -milliseconds]
    set old_time [expr {$end - $start}]
    
    # Test new syntax performance
    set start [clock clicks -milliseconds]
    for {set i 0} {$i < $iterations} {incr i} {
        $COMMAND_OLD -dtype float16 -device_type cuda
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

# Test 23: Comprehensive workflow test
puts "Test 23: Comprehensive workflow test..."
if {[catch {
    # Test complete autocast dtype workflow
    
    # 1. Enable autocast with initial dtype
    torch::autocast_enable cuda float16
    set initial_enabled [torch::autocast_is_enabled cuda]
    
    # 2. Change dtype using positional syntax
    set result_pos [$COMMAND_OLD bfloat16 cuda]
    
    # 3. Change dtype using named syntax
    set result_named [$COMMAND_OLD -dtype float32 -device_type cuda]
    
    # 4. Change dtype using camelCase alias
    set result_camel [$COMMAND_NEW -data_type float16 -device cuda]
    
    # 5. Verify autocast is still enabled
    set final_enabled [torch::autocast_is_enabled cuda]
    
    # Verify workflow
    if {$initial_enabled == 1 && $final_enabled == 1 && 
        $result_pos eq "autocast dtype set" && 
        $result_named eq "autocast dtype set" && 
        $result_camel eq "autocast dtype set"} {
        puts "  Comprehensive workflow: OK"
        puts "    Initial enabled: $initial_enabled"
        puts "    Final enabled: $final_enabled"
        puts "    All set operations successful"
    } else {
        puts "  ❌ Comprehensive workflow failed"
        exit 1
    }
} result]} {
    puts "  ❌ Comprehensive workflow test failed: $result"
    exit 1
}

# Test 24: Independent device management
puts "Test 24: Independent device management..."
if {[catch {
    # Set different dtypes for different devices
    set cuda_result [$COMMAND_OLD -dtype float16 -device cuda]
    set cpu_result [$COMMAND_OLD -dtype bfloat16 -device cpu]
    
    if {$cuda_result eq "autocast dtype set" && $cpu_result eq "autocast dtype set"} {
        puts "  Independent device management: OK"
    } else {
        puts "  ❌ Independent device management failed"
        exit 1
    }
} result]} {
    puts "  ❌ Independent device management test failed: $result"
    exit 1
}

# Test 25: Edge case - rapid dtype changes
puts "Test 25: Edge case - rapid dtype changes..."
if {[catch {
    set all_correct 1
    set dtypes [list "float16" "bfloat16" "float32"]
    
    for {set i 0} {$i < 10} {incr i} {
        set dtype [lindex $dtypes [expr {$i % 3}]]
        set result [$COMMAND_OLD -dtype $dtype -device cuda]
        
        if {$result ne "autocast dtype set"} {
            set all_correct 0
            break
        }
    }
    
    if {$all_correct} {
        puts "  Rapid dtype changes: OK (all changes successful)"
    } else {
        puts "  ❌ Rapid dtype changes failed"
        exit 1
    }
} result]} {
    puts "  ❌ Rapid dtype changes test failed: $result"
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