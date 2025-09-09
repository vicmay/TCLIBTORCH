# tests/refactored/template_test.tcl
# Template test file for refactored commands
# Copy this file and customize for your specific command

package require torch

# Configuration
set COMMAND_NAME "template"  # Change this to your command name
set COMMAND_OLD "torch::template"  # Change this to old command name
set COMMAND_NEW "torch::template"  # Change this to new command name (camelCase if needed)

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    # TODO: Replace with actual test for your command
    # Example: set result [$COMMAND_OLD arg1 arg2 arg3]
    puts "  Original syntax: OK"
} result]} {
    puts "  ❌ Original syntax failed: $result"
    exit 1
}

# Test 2: New named parameter syntax
puts "Test 2: New named parameter syntax..."
if {[catch {
    # TODO: Replace with actual test for your command
    # Example: set result [$COMMAND_NEW -param1 value1 -param2 value2]
    puts "  Named parameters: OK"
} result]} {
    puts "  ❌ Named parameters failed: $result"
    exit 1
}

# Test 3: Mixed syntax (if supported)
puts "Test 3: Mixed syntax..."
if {[catch {
    # TODO: Replace with actual test for your command
    # Example: set result [$COMMAND_NEW arg1 -param2 value2]
    puts "  Mixed syntax: OK"
} result]} {
    puts "  ❌ Mixed syntax failed: $result"
    exit 1
}

# Test 4: Default parameters
puts "Test 4: Default parameters..."
if {[catch {
    # TODO: Replace with actual test for your command
    # Example: set result [$COMMAND_NEW -required_param value]
    puts "  Default parameters: OK"
} result]} {
    puts "  ❌ Default parameters failed: $result"
    exit 1
}

# Test 5: Error handling - invalid parameter names
puts "Test 5: Error handling - invalid parameter names..."
if {[catch {
    # TODO: Replace with actual test for your command
    # Example: $COMMAND_NEW -invalid_param value
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 6: Error handling - missing required parameters
puts "Test 6: Error handling - missing required parameters..."
if {[catch {
    # TODO: Replace with actual test for your command
    # Example: $COMMAND_NEW -optional_param value  # Missing required param
    puts "  ❌ Should have failed with missing required parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 7: Error handling - invalid parameter values
puts "Test 7: Error handling - invalid parameter values..."
if {[catch {
    # TODO: Replace with actual test for your command
    # Example: $COMMAND_NEW -param invalid_value
    puts "  ❌ Should have failed with invalid value"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 8: Performance comparison
puts "Test 8: Performance comparison..."
set iterations 1000
puts "  Running $iterations iterations..."

# Test old syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    # TODO: Replace with actual test for your command
    # Example: $COMMAND_OLD arg1 arg2 arg3
}
set end [clock clicks -milliseconds]
set old_time [expr {$end - $start}]

# Test new syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    # TODO: Replace with actual test for your command
    # Example: $COMMAND_NEW -param1 value1 -param2 value2
}
set end [clock clicks -milliseconds]
set new_time [expr {$end - $start}]

puts "  Old syntax: ${old_time}ms"
puts "  New syntax: ${new_time}ms"
puts "  Performance: OK (within acceptable range)"

# Test 9: Integration with other commands
puts "Test 9: Integration with other commands..."
if {[catch {
    # TODO: Replace with actual integration test
    # Example: Create tensor, use your command, verify result
    puts "  Integration: OK"
} result]} {
    puts "  ❌ Integration failed: $result"
    exit 1
}

# Test 10: CUDA operations (if applicable)
if {[torch::cuda_is_available]} {
    puts "Test 10: CUDA operations..."
    if {[catch {
        # TODO: Replace with actual CUDA test for your command
        # Example: $COMMAND_NEW -device cuda -param value
        puts "  CUDA operations: OK"
    } result]} {
        puts "  ❌ CUDA operations failed: $result"
        exit 1
    }
} else {
    puts "Test 10: CUDA operations... (skipped - CUDA not available)"
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Update tracking: mark command complete in COMMAND-TRACKING.md"
puts "   3. Commit changes: ./scripts/commit_refactored.sh $COMMAND_NAME"
puts "   4. Move to next command: ./scripts/select_next_command.sh" 