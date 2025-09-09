#!/bin/bash
# scripts/commit_refactored.sh [command_name] [batch_commands]
# Commits refactored commands with proper git practices

if [ $# -eq 0 ]; then
    echo "Usage: $0 command_name"
    echo "   or: $0 batch 'command1,command2,command3'"
    echo ""
    echo "Examples:"
    echo "  $0 zeros"
    echo "  $0 batch 'zeros,ones,tensor_create'"
    exit 1
fi

COMMIT_TYPE=$1
COMMANDS=$2

echo "=== Committing Refactored Commands ==="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository!"
    exit 1
fi

# Check if there are any changes to commit
if git diff --cached --quiet && git diff --quiet; then
    echo "❌ No changes to commit!"
    echo "   Make sure you've implemented, tested, and documented your refactored command(s)"
    exit 1
fi

# Single command commit
if [ "$COMMIT_TYPE" != "batch" ]; then
    COMMAND_NAME=$COMMIT_TYPE
    
    echo "📝 Committing single command: torch::$COMMAND_NAME"
    echo ""
    
    # Stage all changes
    echo "🔧 Staging all changes..."
    git add .
    
    # Create commit message
    COMMIT_MSG="Refactor: $COMMAND_NAME with named parameters"
    
    echo "💾 Committing changes..."
    echo "   Message: $COMMIT_MSG"
    echo ""
    
    if git commit -am "$COMMIT_MSG"; then
        echo "✅ Successfully committed torch::$COMMAND_NAME"
        echo ""
        echo "📋 Next steps:"
        echo "   1. Update tracking: mark command complete in COMMAND-TRACKING.md"
        echo "   2. Push branch: git push origin refactor/$COMMAND_NAME"
        echo "   3. Move to next command: ./scripts/select_next_command.sh"
    else
        echo "❌ Commit failed!"
        exit 1
    fi

# Batch command commit
else
    if [ -z "$COMMANDS" ]; then
        echo "❌ Batch mode requires list of commands!"
        echo "   Example: $0 batch 'zeros,ones,tensor_create'"
        exit 1
    fi
    
    echo "📝 Committing batch of commands: $COMMANDS"
    echo ""
    
    # Stage all changes
    echo "🔧 Staging all changes..."
    git add .
    
    # Create commit message
    COMMIT_MSG="Refactor: batch of commands with named parameters ($COMMANDS)"
    
    echo "💾 Committing changes..."
    echo "   Message: $COMMIT_MSG"
    echo ""
    
    if git commit -am "$COMMIT_MSG"; then
        echo "✅ Successfully committed batch: $COMMANDS"
        echo ""
        echo "📋 Next steps:"
        echo "   1. Update tracking: mark all commands complete in COMMAND-TRACKING.md"
        echo "   2. Push branch: git push origin refactor/batch-$(date +%Y%m%d)"
        echo "   3. Check progress: ./scripts/update_progress.sh"
    else
        echo "❌ Commit failed!"
        exit 1
    fi
fi

echo ""
echo "📊 Current git status:"
git status --short

echo ""
echo "📈 Progress update:"
./scripts/update_progress.sh 