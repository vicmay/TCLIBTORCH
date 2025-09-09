# tests/refactored/eye_test.tcl
if {[catch {load ./libtorchtcl.so} res]} { puts "warn: $res" }
if {![llength [info commands torch::eye]]} { puts "❌ torch::eye missing"; exit 1 }
set CMD "torch::eye"
puts "=== Testing $CMD ==="
puts "Positional..."; if {[catch { set h1 [$CMD 3] } r]} { puts "❌ $r"; exit 1 }
puts "Rectangular..."; if {[catch { set h2 [$CMD 3 4 float32 cpu false] } r]} { puts "❌ $r"; exit 1 }
puts "Named..."; if {[catch { set h3 [$CMD -n 3 -m 4 -dtype float32 -device cpu -requiresGrad false] } r]} { puts "❌ $r"; exit 1 }
puts "Named default m..."; if {[catch { set h4 [$CMD -n 5] } r]} { puts "❌ $r"; exit 1 }
puts "Missing n..."; if {![catch { $CMD -m 3 } r]} { puts "❌ should fail"; exit 1 } else { puts "OK - $r" }
puts "Invalid dtype..."; if {![catch { $CMD -n 3 -dtype bad } r]} { puts "❌ should fail"; exit 1 } else { puts "OK - $r" }
puts "✅ eye tests passed" 