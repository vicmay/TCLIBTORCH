#!/usr/bin/env tclsh

# Load the library
load ./libtorchtcl.so

puts "=== Testing Memory and Performance Operations ==="

# Test memory_stats
puts "\n1. Testing torch::memory_stats"
try {
    set result [torch::memory_stats]
    puts "torch::memory_stats: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test memory_summary
puts "\n2. Testing torch::memory_summary"
try {
    set result [torch::memory_summary]
    puts "torch::memory_summary: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test memory_snapshot
puts "\n3. Testing torch::memory_snapshot"
try {
    set result [torch::memory_snapshot]
    puts "torch::memory_snapshot: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test empty_cache
puts "\n4. Testing torch::empty_cache"
try {
    set result [torch::empty_cache]
    puts "torch::empty_cache: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test synchronize
puts "\n5. Testing torch::synchronize"
try {
    set result [torch::synchronize]
    puts "torch::synchronize: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test profiler_start
puts "\n6. Testing torch::profiler_start"
try {
    set result [torch::profiler_start]
    puts "torch::profiler_start: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test profiler_stop
puts "\n7. Testing torch::profiler_stop"
try {
    set result [torch::profiler_stop]
    puts "torch::profiler_stop: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test benchmark
puts "\n8. Testing torch::benchmark"
try {
    set result [torch::benchmark "matrix_multiply"]
    puts "torch::benchmark: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test set_flush_denormal
puts "\n9. Testing torch::set_flush_denormal"
try {
    set result [torch::set_flush_denormal 1]
    puts "torch::set_flush_denormal true: $result"
    set result [torch::set_flush_denormal 0]
    puts "torch::set_flush_denormal false: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test get_num_threads
puts "\n10. Testing torch::get_num_threads"
try {
    set result [torch::get_num_threads]
    puts "torch::get_num_threads: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test set_num_threads
puts "\n11. Testing torch::set_num_threads"
try {
    set original_threads [torch::get_num_threads]
    set result [torch::set_num_threads 4]
    puts "torch::set_num_threads 4: $result"
    set current_threads [torch::get_num_threads]
    puts "Current threads after setting to 4: $current_threads"
    
    # Restore original
    set result [torch::set_num_threads $original_threads]
    puts "Restored threads to: $original_threads"
} on error {err} {
    puts "ERROR: $err"
}

puts "\n=== Memory and Performance Operations Testing Complete ===" 