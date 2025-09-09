#!/usr/bin/env tclsh

# Load the LibTorch TCL extension
if {[catch {load ./libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    exit 1
}

puts "Testing LibTorch TCL Padding Layers - Basic Functionality"
puts "=========================================================="

# Count new commands to verify they were added
set command_count [llength [info commands torch::*]]
puts "Current command count: $command_count"

# Test reflection padding with proper tensor dimensions
puts "\n=== Testing Reflection Padding ==="

# 1D input needs to be at least 2D for reflection_pad1d (batch, length)
puts "Testing Reflection Pad 1D with 2D tensor..."
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set x [torch::tensor_reshape $x {1 4}]
puts "Input tensor shape: [torch::tensor_shape $x]"
if {[catch {torch::reflection_pad1d $x {1 1}} result]} {
    puts "Error: $result"
} else {
    puts "Output tensor shape: [torch::tensor_shape $result]"
    puts "Success!"
}

# 2D input needs to be at least 3D for reflection_pad2d (batch, height, width)  
puts "\nTesting Reflection Pad 2D with 3D tensor..."
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set x [torch::tensor_reshape $x {1 2 2}]
puts "Input tensor shape: [torch::tensor_shape $x]"
if {[catch {torch::reflection_pad2d $x {1 1 1 1}} result]} {
    puts "Error: $result"
} else {
    puts "Output tensor shape: [torch::tensor_shape $result]"
    puts "Success!"
}

# Test replication padding
puts "\n=== Testing Replication Padding ==="

puts "Testing Replication Pad 1D with 2D tensor..."
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set x [torch::tensor_reshape $x {1 4}]
if {[catch {torch::replication_pad1d $x {1 1}} result]} {
    puts "Error: $result"
} else {
    puts "Output tensor shape: [torch::tensor_shape $result]"
    puts "Success!"
}

# Test constant padding  
puts "\n=== Testing Constant Padding ==="

puts "Testing Constant Pad 1D..."
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
if {[catch {torch::constant_pad1d $x {1 1} 9.0} result]} {
    puts "Error: $result"
} else {
    puts "Output tensor shape: [torch::tensor_shape $result]"
    puts "Success!"
}

# Test zero padding
puts "\n=== Testing Zero Padding ==="

puts "Testing Zero Pad 1D..."
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
if {[catch {torch::zero_pad1d $x {1 1}} result]} {
    puts "Error: $result"
} else {
    puts "Output tensor shape: [torch::tensor_shape $result]"
    puts "Success!"
}

# Test circular padding with higher dimensional tensor (add batch and channel dims)
puts "\n=== Testing Circular Padding ==="

puts "Testing Circular Pad 2D with 4D tensor (batch, channels, height, width)..."
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set x [torch::tensor_reshape $x {1 1 2 2}]
puts "Input tensor shape: [torch::tensor_shape $x]"
if {[catch {torch::circular_pad2d $x {1 1 1 1}} result]} {
    puts "Error: $result"
} else {
    puts "Output tensor shape: [torch::tensor_shape $result]"
    puts "Success!"
}

# Verify command count increased by 15
set new_command_count [llength [info commands torch::*]]
set added_commands [expr $new_command_count - $command_count]

puts "\n=========================================="
puts "Summary:"
puts "Commands before: $command_count"
puts "Commands after: $new_command_count"
puts "Added commands: $added_commands"

if {$added_commands >= 15} {
    puts "🎉 SUCCESS: 15 padding operations added!"
    puts "New padding commands available:"
    puts "  - torch::reflection_pad1d, torch::reflection_pad2d, torch::reflection_pad3d"
    puts "  - torch::replication_pad1d, torch::replication_pad2d, torch::replication_pad3d"
    puts "  - torch::constant_pad1d, torch::constant_pad2d, torch::constant_pad3d"
    puts "  - torch::circular_pad1d, torch::circular_pad2d, torch::circular_pad3d"
    puts "  - torch::zero_pad1d, torch::zero_pad2d, torch::zero_pad3d"
    exit 0
} else {
    puts "❌ ERROR: Expected 15 new commands, but only $added_commands were added"
    exit 1
} 