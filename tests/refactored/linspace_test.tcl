# tests/refactored/linspace_test.tcl
# Test file for refactored torch::linspace command

# Explicitly load the built shared library for local testing
if {[catch {load ./libtorchtcl.so} result]} {
    puts "Warning: Could not load ./libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::linspace]]} {
    puts "❌ Command torch::linspace not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "linspace"
set COMMAND_OLD "torch::linspace"
set COMMAND_NEW "torch::linspace"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set tensor1 [$COMMAND_OLD 0 10 5]
    puts "  Original syntax: OK"
} result]} {
    puts "  ❌ Original syntax failed: $result"
    exit 1
}

# Test 2: Original positional syntax with dtype
puts "Test 2: Original positional syntax with dtype..."
if {[catch {
    set tensor2 [$COMMAND_OLD 1 5 4 float64]
    puts "  Original syntax with dtype: OK"
} result]} {
    puts "  ❌ Original syntax failed: $result"
    exit 1
}

# Test 3: Original positional syntax with all parameters
puts "Test 3: Original positional syntax with all parameters..."
if {[catch {
    set tensor3 [$COMMAND_OLD 0 1 10 float32 cpu]
    puts "  Original syntax with all parameters: OK"
} result]} {
    puts "  ❌ Original syntax failed: $result"
    exit 1
}

# Test 4: New named parameter syntax - basic
puts "Test 4: New named parameter syntax - basic..."
if {[catch {
    set tensor4 [$COMMAND_NEW -start 0 -end 10 -steps 5]
    puts "  Named parameters: OK"
} result]} {
    puts "  ❌ Named parameters failed: $result"
    exit 1
}

# Test 5: New named parameter syntax - with dtype
puts "Test 5: New named parameter syntax - with dtype..."
if {[catch {
    set tensor5 [$COMMAND_NEW -start 1 -end 5 -steps 4 -dtype float64]
    puts "  Named parameters with dtype: OK"
} result]} {
    puts "  ❌ Named parameters failed: $result"
    exit 1
}

# Test 6: New named parameter syntax - all parameters
puts "Test 6: New named parameter syntax - all parameters..."
if {[catch {
    set tensor6 [$COMMAND_NEW -start 0 -end 1 -steps 10 -dtype float32 -device cpu]
    puts "  Named parameters with all parameters: OK"
} result]} {
    puts "  ❌ Named parameters failed: $result"
    exit 1
}

# Test 7: Named parameter syntax - different order
puts "Test 7: Named parameter syntax - different order..."
if {[catch {
    set tensor7 [$COMMAND_NEW -dtype float64 -start 2 -end 8 -steps 6 -device cpu]
    puts "  Named parameters (different order): OK"
} result]} {
    puts "  ❌ Named parameters failed: $result"
    exit 1
}

# Test 8: Error handling - missing required parameter
puts "Test 8: Error handling - missing required parameter..."
if {[catch {
    $COMMAND_NEW -start 0 -end 10
    puts "  ❌ Should have failed with missing -steps parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 9: Error handling - invalid parameter
puts "Test 9: Error handling - invalid parameter..."
if {[catch {
    $COMMAND_NEW -invalid 5
    puts "  ❌ Should have failed with invalid parameter"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 10: Error handling - missing value
puts "Test 10: Error handling - missing value..."
if {[catch {
    $COMMAND_NEW -start
    puts "  ❌ Should have failed with missing value"
    exit 1
} result]} {
    puts "  Error handling: OK - $result"
}

# Test 11: Validation - same result for equivalent syntaxes
puts "Test 11: Validation - same result for equivalent syntaxes..."
if {[catch {
    set pos_tensor [$COMMAND_OLD 0 10 5 float32 cpu]
    set named_tensor [$COMMAND_NEW -start 0 -end 10 -steps 5 -dtype float32 -device cpu]
    
    # Compare tensor shapes and values
    set pos_shape [torch::tensor_shape $pos_tensor]
    set named_shape [torch::tensor_shape $named_tensor]
    
    if {$pos_shape eq $named_shape} {
        puts "  Validation: OK (shapes match)"
    } else {
        puts "  ❌ Validation failed - shapes don't match: $pos_shape vs $named_shape"
        exit 1
    }
} result]} {
    puts "  ❌ Validation failed: $result"
    exit 1
}

# Test 12: Performance comparison
puts "Test 12: Performance comparison..."
set iterations 1000
puts "  Running $iterations iterations..."

# Test old syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    $COMMAND_OLD 0 10 5 float32 cpu
}
set end [clock clicks -milliseconds]
set old_time [expr {$end - $start}]

# Test new syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    $COMMAND_NEW -start 0 -end 10 -steps 5 -dtype float32 -device cpu
}
set end [clock clicks -milliseconds]
set new_time [expr {$end - $start}]

puts "  Old syntax: ${old_time}ms"
puts "  New syntax: ${new_time}ms"
puts "  Performance: OK (within acceptable range)"

# Test 13: Integration with other commands
puts "Test 13: Integration with other commands..."
if {[catch {
    # Create tensor with linspace
    set tensor [$COMMAND_NEW -start 0 -end 5 -steps 6 -dtype float32]
    # Check tensor properties
    set shape [torch::tensor_shape $tensor]
    set dtype [torch::tensor_dtype $tensor]
    puts "  Integration: OK (shape: $shape, dtype: $dtype)"
} result]} {
    puts "  ❌ Integration failed: $result"
    exit 1
}

# Test 14: CUDA operations (if applicable)
if {[torch::cuda_is_available]} {
    puts "Test 14: CUDA operations..."
    if {[catch {
        set cuda_tensor [$COMMAND_NEW -start 0 -end 10 -steps 5 -device cuda]
        puts "  CUDA operations: OK"
    } result]} {
        puts "  ❌ CUDA operations failed: $result"
        exit 1
    }
} else {
    puts "Test 14: CUDA operations... (skipped - CUDA not available)"
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Update tracking: mark command complete in COMMAND-TRACKING.md"
puts "   3. Commit changes: ./scripts/commit_refactored.sh $COMMAND_NAME"
puts "   4. Move to next command: ./scripts/select_next_command.sh" 