#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create test waveform
proc create_test_waveform {} {
    # Create a simple sine wave
    set freq 440.0  ;# Hz
    set duration 1.0  ;# seconds
    set sample_rate 44100.0
    set t [torch::arange [expr {$duration * $sample_rate}] float32]
    set sr_tensor [torch::tensor_create $sample_rate float32]
    set t [torch::tensor_div $t $sr_tensor]
    set omega [expr {2.0 * 3.14159 * $freq}]
    set omega_tensor [torch::tensor_create $omega float32]
    set t [torch::tensor_mul $t $omega_tensor]
    set waveform [torch::sin $t]
    return $waveform
}

# Test cases for positional syntax
test pitch-1.1 {Basic positional syntax - shift up one octave} {
    set waveform [create_test_waveform]
    set result [torch::pitch_shift $waveform 44100.0 12.0]
    expr {[llength [torch::tensor_to_list $result]] > 0}
} {1}

test pitch-1.2 {Basic positional syntax - shift down one octave} {
    set waveform [create_test_waveform]
    set result [torch::pitch_shift $waveform 44100.0 -12.0]
    expr {[llength [torch::tensor_to_list $result]] > 0}
} {1}

# Test cases for named syntax
test pitch-2.1 {Named parameter syntax - shift up one octave} {
    set waveform [create_test_waveform]
    set result [torch::pitch_shift -waveform $waveform -sampleRate 44100.0 -nSteps 12.0]
    expr {[llength [torch::tensor_to_list $result]] > 0}
} {1}

test pitch-2.2 {Named parameter syntax - shift down one octave} {
    set waveform [create_test_waveform]
    set result [torch::pitch_shift -waveform $waveform -sampleRate 44100.0 -nSteps -12.0]
    expr {[llength [torch::tensor_to_list $result]] > 0}
} {1}

# Test cases for camelCase alias
test pitch-3.1 {CamelCase alias - positional syntax} {
    set waveform [create_test_waveform]
    set result [torch::pitchShift $waveform 44100.0 12.0]
    expr {[llength [torch::tensor_to_list $result]] > 0}
} {1}

test pitch-3.2 {CamelCase alias - named syntax} {
    set waveform [create_test_waveform]
    set result [torch::pitchShift -waveform $waveform -sampleRate 44100.0 -nSteps 12.0]
    expr {[llength [torch::tensor_to_list $result]] > 0}
} {1}

# Error handling tests
test pitch-4.1 {Missing required parameters} {
    catch {torch::pitch_shift} result
    set result
} {Error in pitch_shift: Usage: torch::pitch_shift waveform sample_rate n_steps | torch::pitch_shift -waveform tensor -sampleRate value -nSteps value}

test pitch-4.2 {Invalid waveform tensor} {
    catch {torch::pitch_shift invalid_tensor 44100.0 12.0} result
    set result
} {Error in pitch_shift: Invalid tensor}

test pitch-4.3 {Invalid sample rate} {
    set waveform [create_test_waveform]
    catch {torch::pitch_shift $waveform -1.0 12.0} result
    set result
} {Error in pitch_shift: Required parameters missing or invalid (waveform and sample_rate required)}

cleanupTests 