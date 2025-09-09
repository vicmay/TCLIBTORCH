#!/bin/bash
# scripts/test_refactored.sh command_name
# Runs tests for a specific refactored command

COMMAND=$1
if [ -z "$COMMAND" ]; then
    echo "Usage: $0 command_name"
    echo ""
    echo "Examples:"
    echo "  $0 zeros"
    echo "  $0 tensor_create"
    echo "  $0 tensor_add"
    exit 1
fi

echo "=== Testing Refactored Command: torch::$COMMAND ==="
echo ""

# Check if test file exists
TEST_FILE="tests/refactored/${COMMAND}_test.tcl"
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    echo ""
    echo "💡 Create the test file first:"
    echo "   cp tests/refactored/template_test.tcl $TEST_FILE"
    echo "   # Then edit the test file for your command"
    exit 1
fi

# Build the project
echo "🔨 Building project..."
if ! ./build.sh > /dev/null 2>&1; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build successful"
echo ""

# Run the test
echo "🧪 Running tests for torch::$COMMAND..."
echo "Test file: $TEST_FILE"
echo ""

if tclsh "$TEST_FILE"; then
    echo ""
    echo "✅ All tests passed for torch::$COMMAND"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Review test output above"
    echo "   2. Create documentation: docs/refactored/${COMMAND}.md"
    echo "   3. Update tracking: mark command complete in COMMAND-TRACKING.md"
    echo "   4. Commit changes: ./scripts/commit_refactored.sh $COMMAND"
    echo "   5. Move to next command: ./scripts/select_next_command.sh"
else
    echo ""
    echo "❌ Tests failed for torch::$COMMAND"
    echo ""
    echo "🔧 Debugging tips:"
    echo "   1. Check the test file: $TEST_FILE"
    echo "   2. Verify command implementation in src/libtorchtcl.cpp"
    echo "   3. Check for compilation errors in build output"
    echo "   4. Run individual test cases manually"
    exit 1
fi 