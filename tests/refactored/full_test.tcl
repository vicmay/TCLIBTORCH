# tests/refactored/full_test.tcl
if {[catch {load ./libtorchtcl.so} result]} { puts "Warn: $result" }
if {![llength [info commands torch::full]]} { puts "❌ torch::full missing"; exit 1 }
set CMD "torch::full"
puts "=== Testing $CMD ==="
puts "Positional..."
if {[catch { set h1 [$CMD {2 2} 5.0 float32 cpu false] } res]} { puts "❌ $res"; exit 1 }
puts "Named..."
if {[catch { set h2 [$CMD -shape {2 2} -value 5.0 -dtype float32 -device cpu -requiresGrad false] } res]} { puts "❌ $res"; exit 1 }
puts "Missing required param..."
if {![catch { $CMD -shape {2 2} -dtype float32 } res]} { puts "❌ should fail"; exit 1 } else { puts "OK - $res" }
puts "Invalid dtype..."
if {![catch { $CMD -shape {2 2} -value 1 -dtype bad } res]} { puts "❌ should fail"; exit 1 } else { puts "OK - $res" }
puts "✅ full tests passed" 