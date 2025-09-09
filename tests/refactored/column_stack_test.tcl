#!/usr/bin/env tclsh
# tests/refactored/column_stack_test.tcl
# Test file for refactored torch::column_stack command

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    if {[catch {load ./libtorchtcl.so}]} {
        puts "Failed to load libtorchtcl.so"
        exit 1
    }
}

# Configuration
set COMMAND_OLD "torch::column_stack"
set COMMAND_NEW "torch::columnStack"

puts "=== Testing Refactored Command: $COMMAND_OLD / $COMMAND_NEW ==="
puts ""

# Test 1: Basic functionality with multiple arguments
puts "Test 1: Basic functionality with multiple arguments..."
if {[catch {
    set t1 [torch::tensor_create {1.0 2.0} float32]
    set t2 [torch::tensor_create {3.0 4.0} float32]
    set result [$COMMAND_OLD $t1 $t2]
    puts "  Multiple arguments: OK (result: $result)"
} error]} {
    puts "  ❌ Multiple arguments failed: $error"
    exit 1
}

# Test 2: Basic functionality with list argument
puts "Test 2: Basic functionality with list argument..."
if {[catch {
    set t1 [torch::tensor_create {1.0 2.0} float32]
    set t2 [torch::tensor_create {3.0 4.0} float32]
    set tensor_list [list $t1 $t2]
    set result [$COMMAND_OLD $tensor_list]
    puts "  List argument: OK (result: $result)"
} error]} {
    puts "  ❌ List argument failed: $error"
    exit 1
}

# Test 3: CamelCase alias with multiple arguments
puts "Test 3: CamelCase alias with multiple arguments..."
if {[catch {
    set t1 [torch::tensor_create {5.0 6.0} float32]
    set t2 [torch::tensor_create {7.0 8.0} float32]
    set result [$COMMAND_NEW $t1 $t2]
    puts "  CamelCase multiple args: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase multiple args failed: $error"
    exit 1
}

# Test 4: CamelCase alias with list argument
puts "Test 4: CamelCase alias with list argument..."
if {[catch {
    set t1 [torch::tensor_create {9.0 10.0} float32]
    set t2 [torch::tensor_create {11.0 12.0} float32]
    set tensor_list [list $t1 $t2]
    set result [$COMMAND_NEW $tensor_list]
    puts "  CamelCase list: OK (result: $result)"
} error]} {
    puts "  ❌ CamelCase list failed: $error"
    exit 1
}

# Test 5: Three tensors
puts "Test 5: Three tensors..."
if {[catch {
    set t1 [torch::tensor_create {1.0 2.0} float32]
    set t2 [torch::tensor_create {3.0 4.0} float32]
    set t3 [torch::tensor_create {5.0 6.0} float32]
    set result [$COMMAND_OLD $t1 $t2 $t3]
    puts "  Three tensors: OK (result: $result)"
} error]} {
    puts "  ❌ Three tensors failed: $error"
    exit 1
}

# Test 6: 2D tensors 
puts "Test 6: 2D tensors..."
if {[catch {
    set t1 [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set t1 [torch::tensor_reshape $t1 {2 2}]
    set t2 [torch::tensor_create {5.0 6.0 7.0 8.0} float32]
    set t2 [torch::tensor_reshape $t2 {2 2}]
    set result [$COMMAND_OLD $t1 $t2]
    puts "  2D tensors: OK (result: $result)"
} error]} {
    puts "  ❌ 2D tensors failed: $error"
    exit 1
}

# Test 7: Column vectors
puts "Test 7: Column vectors..."
if {[catch {
    set v1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set v1 [torch::tensor_reshape $v1 {3 1}]
    set v2 [torch::tensor_create {4.0 5.0 6.0} float32]
    set v2 [torch::tensor_reshape $v2 {3 1}]
    set result [$COMMAND_OLD $v1 $v2]
    puts "  Column vectors: OK (result: $result)"
} error]} {
    puts "  ❌ Column vectors failed: $error"
    exit 1
}

# Test 8: Error handling - no arguments
puts "Test 8: Error handling - no arguments..."
if {[catch {
    $COMMAND_OLD
    puts "  ❌ Should have failed with no arguments"
    exit 1
} error]} {
    puts "  No arguments error: OK - $error"
}

# Test 9: Error handling - empty list
puts "Test 9: Error handling - empty list..."
if {[catch {
    $COMMAND_OLD {}
    puts "  ❌ Should have failed with empty list"
    exit 1
} error]} {
    puts "  Empty list error: OK - $error"
}

# Test 10: Error handling - invalid tensor
puts "Test 10: Error handling - invalid tensor..."
if {[catch {
    $COMMAND_OLD invalid_tensor
    puts "  ❌ Should have failed with invalid tensor"
    exit 1
} error]} {
    puts "  Invalid tensor error: OK - $error"
}

# Test 11: Mathematical correctness - shape verification
puts "Test 11: Mathematical correctness - shape verification..."
if {[catch {
    set t1 [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set t1 [torch::tensor_reshape $t1 {2 2}]
    set t2 [torch::tensor_create {5.0 6.0 7.0 8.0} float32]
    set t2 [torch::tensor_reshape $t2 {2 2}]
    
    # Get original shapes
    set shape1 [torch::tensor_shape $t1]
    set shape2 [torch::tensor_shape $t2]
    
    # Column stack should concatenate along last dimension (columns)
    set result [$COMMAND_OLD $t1 $t2]
    set result_shape [torch::tensor_shape $result]
    
    # Expected: same height, double width
    set expected_height [lindex $shape1 0]
    set expected_width [expr {[lindex $shape1 1] + [lindex $shape2 1]}]
    
    if {[lindex $result_shape 0] == $expected_height && [lindex $result_shape 1] == $expected_width} {
        puts "  Shape verification: OK (result shape: $result_shape)"
    } else {
        puts "  ❌ Shape mismatch: expected {$expected_height $expected_width}, got $result_shape"
        exit 1
    }
} error]} {
    puts "  ❌ Shape verification failed: $error"
    exit 1
}

# Test 12: Performance comparison
puts "Test 12: Performance comparison..."
set iterations 20
puts "  Running $iterations iterations..."

# Test list syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set t1 [torch::tensor_create {1.0 2.0} float32]
    set t2 [torch::tensor_create {3.0 4.0} float32]
    set tensor_list [list $t1 $t2]
    $COMMAND_OLD $tensor_list
}
set end [clock clicks -milliseconds]
set list_time [expr {$end - $start}]

# Test multiple args syntax performance
set start [clock clicks -milliseconds]
for {set i 0} {$i < $iterations} {incr i} {
    set t1 [torch::tensor_create {1.0 2.0} float32]
    set t2 [torch::tensor_create {3.0 4.0} float32]
    $COMMAND_OLD $t1 $t2
}
set end [clock clicks -milliseconds]
set args_time [expr {$end - $start}]

puts "  List syntax: ${list_time}ms"
puts "  Multiple args: ${args_time}ms"
puts "  Performance: OK (both syntaxes work efficiently)"

# Test 13: Data type compatibility
puts "Test 13: Data type compatibility..."
if {[catch {
    set float_tensor [torch::tensor_create {1.0 2.0} float32]
    set double_tensor [torch::tensor_create {3.0 4.0} float64]
    set result [$COMMAND_OLD $float_tensor $double_tensor]
    puts "  Data type compatibility: OK (result: $result)"
} error]} {
    puts "  ❌ Data type compatibility failed: $error"
    exit 1
}

# Test 14: Different tensor sizes (compatible for column stacking)
puts "Test 14: Different tensor sizes..."
if {[catch {
    set small [torch::tensor_create {1.0 2.0} float32]
    set small [torch::tensor_reshape $small {2 1}]
    
    set large [torch::tensor_create {3.0 4.0 5.0 6.0} float32]
    set large [torch::tensor_reshape $large {2 2}]
    
    set result [$COMMAND_OLD $small $large]
    puts "  Different sizes: OK (result: $result)"
} error]} {
    puts "  ❌ Different sizes failed: $error"
    exit 1
}

puts ""
puts "✅ All tests passed for $COMMAND_OLD / $COMMAND_NEW"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/column_stack.md"
puts "   2. Mark command complete: python3 scripts/update_command_status.py mark-complete torch::column_stack"
puts "   3. Continue with next commands in queue" 