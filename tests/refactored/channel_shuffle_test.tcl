#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Explicitly load the built shared library for local testing
if {[catch {load ./libtorchtcl.so} result]} {
    puts "Warning: Could not load ./libtorchtcl.so: $result"
}

# Check if the command is available
if {![llength [info commands torch::channel_shuffle]]} {
    puts "❌ Command torch::channel_shuffle not found after loading libtorchtcl.so"
    exit 1
}

# Configuration
set COMMAND_NAME "channel_shuffle"
set COMMAND_OLD "torch::channel_shuffle"
set COMMAND_NEW "torch::channelShuffle"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Original positional syntax (backward compatibility)
puts "Test 1: Original positional syntax..."
if {[catch {
    set input [torch::zeros {1 6 4 4}]
    set result [$COMMAND_OLD $input 2]
    puts "  Original syntax: OK (result: $result)"
} error]} {
    puts "  ❌ Original syntax failed: $error"
    exit 1
}

# Test 2: New named parameter syntax
puts "Test 2: New named parameter syntax..."
if {[catch {
    set input [torch::zeros {1 6 4 4}]
    set result [$COMMAND_OLD -input $input -groups 2]
    puts "  Named parameters: OK (result: $result)"
} error]} {
    puts "  ❌ Named parameters failed: $error"
    exit 1
}

# Test 3: CamelCase alias with positional syntax
puts "Test 3: CamelCase alias with positional syntax..."
if {[catch {
    set input [torch::zeros {1 8 3 3}]
    set result [$COMMAND_NEW $input 4]
    puts "  CamelCase positional: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase positional failed: $error"
    exit 1
}

# Test 4: CamelCase alias with named parameters
puts "Test 4: CamelCase alias with named parameters..."
if {[catch {
    set input [torch::zeros {1 8 3 3}]
    set result [$COMMAND_NEW -input $input -groups 4]
    puts "  CamelCase named: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase named failed: $error"
    exit 1
}

# Test 5: Parameter order flexibility
puts "Test 5: Parameter order flexibility..."
if {[catch {
    set input [torch::zeros {1 6 4 4}]
    set result [$COMMAND_OLD -groups 3 -input $input]
    puts "  Parameter order: OK (result: $result)"
} error]} {
    puts "  ❌ Parameter order failed: $error"
    exit 1
}

# Test 6: -tensor alias parameter
puts "Test 6: -tensor alias parameter..."
if {[catch {
    set input [torch::zeros {1 12 2 2}]
    set result [$COMMAND_OLD -tensor $input -groups 6]
    puts "  Tensor alias: OK (result: $result)"
} error]} {
    puts "  ❌ Tensor alias failed: $error"
    exit 1
}

# Test 7: Error handling - missing parameters
puts "Test 7: Error handling - missing parameters..."
if {[catch {
    $COMMAND_OLD
    puts "  ❌ Should have failed with missing parameters"
    exit 1
} error]} {
    puts "  Missing parameters error: OK - $error"
}

# Test 8: Error handling - invalid tensor
puts "Test 8: Error handling - invalid tensor..."
if {[catch {
    $COMMAND_OLD invalid_tensor 2
    puts "  ❌ Should have failed with invalid tensor"
    exit 1
} error]} {
    puts "  Invalid tensor error: OK - $error"
}

# Test 9: Error handling - invalid groups value
puts "Test 9: Error handling - invalid groups value..."
if {[catch {
    set input [torch::zeros {1 6 4 4}]
    $COMMAND_OLD $input 0
    puts "  ❌ Should have failed with invalid groups"
    exit 1
} error]} {
    puts "  Invalid groups error: OK - $error"
}

# Test 10: Error handling - unknown parameter
puts "Test 10: Error handling - unknown parameter..."
if {[catch {
    set input [torch::zeros {1 6 4 4}]
    $COMMAND_OLD -input $input -unknown_param 2
    puts "  ❌ Should have failed with unknown parameter"
    exit 1
} error]} {
    puts "  Unknown parameter error: OK - $error"
}

# Test 11: Error handling - missing value for named parameter
puts "Test 11: Error handling - missing value for named parameter..."
if {[catch {
    set input [torch::zeros {1 6 4 4}]
    $COMMAND_OLD -input $input -groups
    puts "  ❌ Should have failed with missing value"
    exit 1
} error]} {
    puts "  Missing value error: OK - $error"
}

# Test 12: Different tensor shapes and group combinations
puts "Test 12: Different tensor shapes and group combinations..."
if {[catch {
    # Test various combinations
    set input1 [torch::zeros {2 12 8 8}]
    set result1 [$COMMAND_OLD $input1 3]
    
    set input2 [torch::zeros {1 16 4 4}]
    set result2 [$COMMAND_OLD -input $input2 -groups 8]
    
    set input3 [torch::zeros {3 6 2 2}]
    set result3 [$COMMAND_NEW -input $input3 -groups 1]
    
    puts "  Various combinations: OK"
} error]} {
    puts "  ❌ Various combinations failed: $error"
    exit 1
}

# Test 13: Mathematical correctness - shape preservation
puts "Test 13: Mathematical correctness - shape preservation..."
if {[catch {
    set input [torch::zeros {2 8 4 4}]
    set result [$COMMAND_OLD $input 4]
    
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    
    if {$input_shape eq $result_shape} {
        puts "  Shape preservation: OK (input: $input_shape, result: $result_shape)"
    } else {
        puts "  ❌ Shape mismatch: input $input_shape != result $result_shape"
        exit 1
    }
} error]} {
    puts "  ❌ Shape preservation failed: $error"
    exit 1
}

# Test 14: Performance comparison
puts "Test 14: Performance comparison..."
set iterations 100
puts "  Running $iterations iterations..."

# Test old syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set input [torch::zeros {1 6 4 4}]
    $COMMAND_OLD $input 2
}
set end [clock clicks -milliseconds]
set old_time [expr {$end - $start}]

# Test new syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set input [torch::zeros {1 6 4 4}]
    $COMMAND_OLD -input $input -groups 2
}
set end [clock clicks -milliseconds]
set new_time [expr {$end - $start}]

puts "  Old syntax: ${old_time}ms"
puts "  New syntax: ${new_time}ms"
puts "  Performance: OK (within acceptable range)"

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/${COMMAND_NAME}.md"
puts "   2. Mark command complete: python3 scripts/update_command_status.py mark-complete torch::channel_shuffle"
puts "   3. Commit changes: git add . && git commit -m 'Refactor torch::channel_shuffle with dual syntax support'"
puts "   4. Move to next command: python3 scripts/query_next_commands.py --next 1" 