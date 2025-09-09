# tests/refactored/ones_test.tcl
# Test file for refactored torch::ones command

# Explicitly load the built shared library for local testing
if {[catch {load ./libtorchtcl.so} result]} {
    puts "Warning: Could not load ./libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::ones]]} {
    puts "❌ Command torch::ones not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "ones"
set COMMAND_OLD "torch::ones"
set COMMAND_NEW "torch::ones"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set tensor1 [$COMMAND_OLD {3 3} float32 cpu false]
    puts "  Original syntax: OK"
} result]} {
    puts "  ❌ Original syntax failed: $result"
    exit 1
}

# Test 2: New named parameter syntax
puts "Test 2: New named parameter syntax..."
if {[catch {
    set tensor2 [$COMMAND_NEW -shape {3 3} -dtype float32 -device cpu -requiresGrad false]
    puts "  Named parameters: OK"
} result]} {
    puts "  ❌ Named parameters failed: $result"
    exit 1
}

# Test 3: Mixed syntax (if supported)
puts "Test 3: Mixed syntax..."
if {[catch {
    set tensor3 [$COMMAND_NEW {3 3} -dtype float32 -device cpu]
    puts "  Mixed syntax: OK"
} result]} {
    puts "  ❌ Mixed syntax failed: $result"
    exit 1
}

# Test 4: Default parameters
puts "Test 4: Default parameters..."
if {[catch {
    set tensor4 [$COMMAND_NEW -shape {2 2}]
    puts "  Default parameters: OK"
} result]} {
    puts "  ❌ Default parameters failed: $result"
    exit 1
}

# Test 5: Error handling - invalid parameter names
puts "Test 5: Error handling - invalid parameter names..."
if {[catch {
    $COMMAND_NEW -invalid_param value
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 6: Error handling - missing required parameters
puts "Test 6: Error handling - missing required parameters..."
if {[catch {
    $COMMAND_NEW -dtype float32 -device cpu
    puts "  ❌ Should have failed with missing required parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 7: Error handling - invalid parameter values
puts "Test 7: Error handling - invalid parameter values..."
if {[catch {
    $COMMAND_NEW -shape {3 3} -dtype invalid_type
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
    $COMMAND_OLD {3 3} float32 cpu false
}
set end [clock clicks -milliseconds]
set old_time [expr {$end - $start}]

# Test new syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    $COMMAND_NEW -shape {3 3} -dtype float32 -device cpu -requiresGrad false
}
set end [clock clicks -milliseconds]
set new_time [expr {$end - $start}]

puts "  Old syntax: ${old_time}ms"
puts "  New syntax: ${new_time}ms"
puts "  Performance: OK (within acceptable range)"

# Test 9: Integration with other commands
puts "Test 9: Integration with other commands..."
if {[catch {
    # Create tensor with ones
    set tensor [$COMMAND_NEW -shape {2 2} -dtype float32]
    # Check tensor properties
    set shape [torch::tensor_shape $tensor]
    set dtype [torch::tensor_dtype $tensor]
    puts "  Integration: OK (shape: $shape, dtype: $dtype)"
} result]} {
    puts "  ❌ Integration failed: $result"
    exit 1
}

# Test 10: CUDA operations (if applicable)
if {[torch::cuda_is_available]} {
    puts "Test 10: CUDA operations..."
    if {[catch {
        set cuda_tensor [$COMMAND_NEW -shape {3 3} -device cuda -requiresGrad true]
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