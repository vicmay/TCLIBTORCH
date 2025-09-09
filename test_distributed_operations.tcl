#!/usr/bin/env tclsh

# Load the library
load ./libtorchtcl.so

puts "=== Testing Distributed Operations ===" 

# List of distributed operation commands to test
set commands {
    torch::distributed_gather
    torch::distributed_scatter
    torch::distributed_reduce_scatter
    torch::distributed_all_to_all
    torch::distributed_send
    torch::distributed_recv
    torch::distributed_isend
    torch::distributed_irecv
    torch::distributed_wait
    torch::distributed_test
}

set count 0
foreach cmd $commands {
    incr count
    puts "\n$count. Testing $cmd"
    if {[info commands $cmd] ne ""} {
        puts "$cmd: Available ✓"
    } else {
        puts "$cmd: NOT Available ✗"
    }
}

puts "\n=== Summary: [llength $commands] Distributed Operations Tested ==="

# Test basic functionality with simple examples
puts "\n=== Functional Tests ==="

# Test gather operation
puts "\n1. Testing torch::distributed_gather functionality"
try {
    set x [torch::tensor {{1 2} {3 4}}]
    set result [torch::distributed_gather $x]
    puts "torch::distributed_gather: Success"
} on error {err} {
    puts "ERROR: $err"
}

# Test scatter operation  
puts "\n2. Testing torch::distributed_scatter functionality"
try {
    set x [torch::tensor {{1 2} {3 4}}]
    set result [torch::distributed_scatter $x]
    puts "torch::distributed_scatter: Success"
} on error {err} {
    puts "ERROR: $err"
}

# Test reduce_scatter operation
puts "\n3. Testing torch::distributed_reduce_scatter functionality"
try {
    set x [torch::tensor {{1 2} {3 4}}]
    set result [torch::distributed_reduce_scatter $x]
    puts "torch::distributed_reduce_scatter: Success"
} on error {err} {
    puts "ERROR: $err"
}

# Test all_to_all operation
puts "\n4. Testing torch::distributed_all_to_all functionality"
try {
    set x [torch::tensor {{1 2} {3 4}}]
    set result [torch::distributed_all_to_all $x]
    puts "torch::distributed_all_to_all: Success"
} on error {err} {
    puts "ERROR: $err"
}

# Test send operation
puts "\n5. Testing torch::distributed_send functionality"
try {
    set x [torch::tensor {{1 2} {3 4}}]
    set result [torch::distributed_send $x 1]
    puts "torch::distributed_send: Success - $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test recv operation
puts "\n6. Testing torch::distributed_recv functionality"
try {
    set result [torch::distributed_recv {2 2} 0]
    puts "torch::distributed_recv: Success"
} on error {err} {
    puts "ERROR: $err"
}

# Test isend operation
puts "\n7. Testing torch::distributed_isend functionality"
try {
    set x [torch::tensor {{1 2} {3 4}}]
    set handle [torch::distributed_isend $x 1]
    puts "torch::distributed_isend: Success - handle: $handle"
} on error {err} {
    puts "ERROR: $err"
}

# Test irecv operation
puts "\n8. Testing torch::distributed_irecv functionality"
try {
    set handle [torch::distributed_irecv {2 2} 0]
    puts "torch::distributed_irecv: Success - handle: $handle"
} on error {err} {
    puts "ERROR: $err"
}

# Test wait operation
puts "\n9. Testing torch::distributed_wait functionality"
try {
    set handle "isend_handle_1"
    set result [torch::distributed_wait $handle]
    puts "torch::distributed_wait: Success - $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test test operation
puts "\n10. Testing torch::distributed_test functionality"
try {
    set handle "isend_handle_1"
    set result [torch::distributed_test $handle]
    puts "torch::distributed_test: Success - $result"
} on error {err} {
    puts "ERROR: $err"
}

puts "\n=== All Distributed Operations Tests Complete ===" 