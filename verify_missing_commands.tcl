#!/usr/bin/env tclsh

# Load the library
load ./libtorchtcl.so

# Get current commands
set current_commands [info commands ::torch::*]

# Commands listed as missing in TODO - let's verify them one by one
set supposed_missing {
    "::torch::manual_seed"
    "::torch::initial_seed"
    "::torch::seed"
    "::torch::get_rng_state"
    "::torch::set_rng_state" 
    "::torch::bernoulli"
    "::torch::multinomial"
    "::torch::normal"
    "::torch::uniform"
    "::torch::exponential"
    "::torch::gamma"
    "::torch::poisson"
    "::torch::grad"
    "::torch::jacobian"
    "::torch::hessian"
    "::torch::vjp"
    "::torch::jvp"
    "::torch::functional_call"
    "::torch::vmap"
    "::torch::grad_check"
    "::torch::grad_check_finite_diff"
    "::torch::enable_grad"
    "::torch::no_grad"
    "::torch::set_grad_enabled"
    "::torch::is_grad_enabled"
    "::torch::memory_stats"
    "::torch::memory_summary"
    "::torch::memory_snapshot"
    "::torch::empty_cache"
    "::torch::synchronize"
    "::torch::profiler_start"
    "::torch::profiler_stop"
    "::torch::benchmark"
    "::torch::set_flush_denormal"
    "::torch::get_num_threads"
    "::torch::set_num_threads"
    "::torch::fftshift"
    "::torch::ifftshift"
    "::torch::hilbert"
    "::torch::bartlett_window"
    "::torch::blackman_window"
    "::torch::hamming_window"
    "::torch::hann_window"
    "::torch::kaiser_window"
    "::torch::spectrogram"
    "::torch::melscale_fbanks"
    "::torch::mfcc"
    "::torch::pitch_shift"
    "::torch::time_stretch"
    "::torch::distributed_gather"
    "::torch::distributed_scatter"
    "::torch::distributed_reduce_scatter"
    "::torch::distributed_all_to_all"
    "::torch::distributed_send"
    "::torch::distributed_recv"
    "::torch::distributed_isend"
    "::torch::distributed_irecv"
    "::torch::distributed_wait"
    "::torch::distributed_test"
    "::torch::block_diag"
    "::torch::broadcast_shapes"
    "::torch::squeeze_multiple"
    "::torch::unsqueeze_multiple"
    "::torch::tensor_split"
    "::torch::hsplit"
    "::torch::vsplit"
    "::torch::dsplit"
    "::torch::column_stack"
    "::torch::row_stack"
    "::torch::dstack"
    "::torch::hstack"
    "::torch::vstack"
}

puts "=== VERIFICATION REPORT ==="
puts "Total current commands: [llength $current_commands]"
puts ""

set actually_missing []
set not_missing []

foreach cmd $supposed_missing {
    if {$cmd in $current_commands} {
        lappend not_missing $cmd
    } else {
        lappend actually_missing $cmd
    }
}

puts "Commands listed as missing but ACTUALLY EXIST ([llength $not_missing]):"
foreach cmd $not_missing {
    puts "  ✅ $cmd"
}

puts ""
puts "Commands that are ACTUALLY MISSING ([llength $actually_missing]):"
foreach cmd $actually_missing {
    puts "  ❌ $cmd"
}

puts ""
puts "=== SUMMARY ==="
puts "Commands to implement: [llength $actually_missing]"
puts "False positives in TODO: [llength $not_missing]" 