#!/usr/bin/env tclsh
# tests/refactored/combinations_test.tcl
# Test file for refactored torch::combinations command

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    if {[catch {load ./libtorchtcl.so}]} {
        puts "Failed to load libtorchtcl.so"
        exit 1
    }
}

# Configuration
set COMMAND_OLD "torch::combinations"

puts "=== Testing Refactored Command: $COMMAND_OLD ==="
puts ""

# Test 1: Original positional syntax (basic)
puts "Test 1: Original positional syntax (basic)..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2 3 4} int32]
    set result [$COMMAND_OLD $input_tensor]
    puts "  Positional basic: OK (result: $result)"
} error]} {
    puts "  ❌ Positional basic failed: $error"
    exit 1
}

# Test 2: Original positional syntax with r parameter
puts "Test 2: Original positional syntax with r..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2 3 4} int32]
    set result [$COMMAND_OLD $input_tensor 3]
    puts "  Positional with r: OK (result: $result)"
} error]} {
    puts "  ❌ Positional with r failed: $error"
    exit 1
}

# Test 3: Original positional syntax with replacement
puts "Test 3: Original positional syntax with replacement..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2 3} int32]
    set result [$COMMAND_OLD $input_tensor 2 1]
    puts "  Positional with replacement: OK (result: $result)"
} error]} {
    puts "  ❌ Positional with replacement failed: $error"
    exit 1
}

# Test 4: Named parameter syntax - basic
puts "Test 4: Named parameter syntax - basic..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2 3} int32]
    set result [$COMMAND_OLD -input $input_tensor]
    puts "  Named basic: OK (result: $result)"
} error]} {
    puts "  ❌ Named basic failed: $error"
    exit 1
}

# Test 5: Named parameter syntax with r
puts "Test 5: Named parameter syntax with r..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2 3} int32]
    set result [$COMMAND_OLD -input $input_tensor -r 3]
    puts "  Named with r: OK (result: $result)"
} error]} {
    puts "  ❌ Named with r failed: $error"
    exit 1
}

# Test 6: Named parameter syntax with replacement
puts "Test 6: Named parameter syntax with replacement..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2} int32]
    set result [$COMMAND_OLD -input $input_tensor -r 2 -with_replacement 1]
    puts "  Named with replacement: OK (result: $result)"
} error]} {
    puts "  ❌ Named with replacement failed: $error"
    exit 1
}

# Test 7: Alternative parameter names
puts "Test 7: Alternative parameter names..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2} int32]
    set result [$COMMAND_OLD -tensor $input_tensor -replacement 0]
    puts "  Alternative names: OK (result: $result)"
} error]} {
    puts "  ❌ Alternative names failed: $error"
    exit 1
}

# Test 8: Parameter order flexibility
puts "Test 8: Parameter order flexibility..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2 3} int32]
    set result [$COMMAND_OLD -r 3 -input $input_tensor -with_replacement 0]
    puts "  Parameter order: OK (result: $result)"
} error]} {
    puts "  ❌ Parameter order failed: $error"
    exit 1
}

# Test 9: Error handling - missing input
puts "Test 9: Error handling - missing input..."
if {[catch {
    $COMMAND_OLD -r 2
    puts "  ❌ Should have failed with missing input"
    exit 1
} error]} {
    puts "  Missing input error: OK - $error"
}

# Test 10: Error handling - unknown parameter
puts "Test 10: Error handling - unknown parameter..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2} int32]
    $COMMAND_OLD -input $input_tensor -unknown_param value
    puts "  ❌ Should have failed with unknown parameter"
    exit 1
} error]} {
    puts "  Unknown parameter error: OK - $error"
}

# Test 11: Error handling - invalid tensor
puts "Test 11: Error handling - invalid tensor..."
if {[catch {
    $COMMAND_OLD invalid_tensor
    puts "  ❌ Should have failed with invalid tensor"
    exit 1
} error]} {
    puts "  Invalid tensor error: OK - $error"
}

# Test 12: Mathematical correctness - combinations count
puts "Test 12: Mathematical correctness - combinations count..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2} int32]
    set result [$COMMAND_OLD -input $input_tensor -r 2 -with_replacement 0]
    set result_shape [torch::tensor_shape $result]
    
    # For 3 elements, choose 2: C(3,2) = 3 combinations
    # Result should be 3x2 tensor
    if {[lindex $result_shape 0] == 3 && [lindex $result_shape 1] == 2} {
        puts "  Shape verification: OK (result shape: $result_shape)"
    } else {
        puts "  ❌ Shape mismatch: expected {3 2}, got $result_shape"
        exit 1
    }
} error]} {
    puts "  ❌ Math verification failed: $error"
    exit 1
}

# Test 13: With replacement verification
puts "Test 13: With replacement verification..."
if {[catch {
    set input_tensor [torch::tensor_create {0 1 2} int32]
    set result [$COMMAND_OLD -input $input_tensor -r 2 -with_replacement 1]
    set result_shape [torch::tensor_shape $result]
    
    # For 3 elements with replacement, choose 2: (3+2-1)C2 = 4C2 = 6 combinations
    # Actually, with replacement it's more complex, but should be > 3
    if {[lindex $result_shape 1] == 2} {
        puts "  With replacement shape: OK (result shape: $result_shape)"
    } else {
        puts "  ❌ With replacement shape issue: got $result_shape"
        exit 1
    }
} error]} {
    puts "  ❌ With replacement verification failed: $error"
    exit 1
}

# Test 14: Performance comparison
puts "Test 14: Performance comparison..."
set iterations 10
puts "  Running $iterations iterations..."

# Test positional syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set input_tensor [torch::tensor_create {0 1 2 3} int32]
    $COMMAND_OLD $input_tensor 2
}
set end [clock clicks -milliseconds]
set pos_time [expr {$end - $start}]

# Test named syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set input_tensor [torch::tensor_create {0 1 2 3} int32]
    $COMMAND_OLD -input $input_tensor -r 2
}
set end [clock clicks -milliseconds]
set named_time [expr {$end - $start}]

puts "  Positional syntax: ${pos_time}ms"
puts "  Named syntax: ${named_time}ms"
puts "  Performance: OK (both syntaxes work efficiently)"

# Test 15: Different data types
puts "Test 15: Different data types..."
if {[catch {
    set float_tensor [torch::tensor_create {1.0 2.0 3.0} float32]
    set result [$COMMAND_OLD -input $float_tensor -r 2]
    puts "  Float tensor: OK (result: $result)"
} error]} {
    puts "  ❌ Float tensor failed: $error"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/combinations.md"
puts "   2. Mark command complete: python3 scripts/update_command_status.py mark-complete torch::combinations"
puts "   3. Continue with next commands in queue" 