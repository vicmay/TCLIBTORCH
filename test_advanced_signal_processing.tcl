#!/usr/bin/env tclsh

# Load the library
load ./libtorchtcl.so

puts "=== Testing Advanced Signal Processing Operations ==="

# List of signal processing commands to test
set commands {
    torch::fftshift
    torch::ifftshift
    torch::hilbert
    torch::bartlett_window
    torch::blackman_window
    torch::hamming_window
    torch::hann_window
    torch::kaiser_window
    torch::spectrogram
    torch::melscale_fbanks
    torch::mfcc
    torch::pitch_shift
    torch::time_stretch
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

puts "\n=== Summary: [llength $commands] Advanced Signal Processing commands tested ==="
puts "Note: Command availability verified. Functional testing requires proper tensor input creation." 