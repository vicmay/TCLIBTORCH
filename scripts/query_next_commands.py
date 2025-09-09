#!/usr/bin/env python3
"""
Query the SQLite tracking database to find next commands to refactor.
This provides accurate, real-time status based on actual codebase analysis.
"""

import sqlite3
import sys
import os
from pathlib import Path

def connect_db(db_path='./refactoring_progress.db'):
    """Connect to the tracking database."""
    if not os.path.exists(db_path):
        print(f"Error: Database '{db_path}' not found. Run init_tracking_db.py first.")
        sys.exit(1)
    return sqlite3.connect(db_path)

def print_summary(conn):
    """Print overall progress summary."""
    cursor = conn.cursor()
    
    # Overall stats
    cursor.execute('SELECT COUNT(*) FROM commands')
    total = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM commands WHERE is_complete = 1')
    complete = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM commands WHERE has_dual_syntax = 1')
    dual_syntax = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM commands WHERE has_camel_case_alias = 1')
    camel_case = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM commands WHERE has_tests = 1')
    tests = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM commands WHERE has_documentation = 1')
    docs = cursor.fetchone()[0]
    
    print("=" * 60)
    print("LIBTORCH TCL REFACTORING PROGRESS")
    print("=" * 60)
    print(f"Total commands: {total}")
    print(f"Complete commands: {complete} ({complete/total*100:.1f}%)")
    print(f"Commands with dual syntax: {dual_syntax} ({dual_syntax/total*100:.1f}%)")
    print(f"Commands with camelCase alias: {camel_case} ({camel_case/total*100:.1f}%)")
    print(f"Commands with tests: {tests} ({tests/total*100:.1f}%)")
    print(f"Commands with documentation: {docs} ({docs/total*100:.1f}%)")
    print()

def print_next_commands(conn, limit=10):
    """Print next commands that need refactoring."""
    cursor = conn.cursor()
    
    # Commands that need refactoring (not complete)
    cursor.execute('''
        SELECT name, category, has_dual_syntax, has_camel_case_alias, 
               has_tests, has_documentation, source_file
        FROM commands 
        WHERE is_complete = 0
        ORDER BY 
            CASE category
                WHEN 'tensor_creation' THEN 1
                WHEN 'basic_operations' THEN 2
                WHEN 'activation_functions' THEN 3
                WHEN 'neural_layers' THEN 4
                WHEN 'optimizers' THEN 5
                WHEN 'loss_functions' THEN 6
                ELSE 7
            END,
            name
        LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    
    if not results:
        print("🎉 ALL COMMANDS ARE COMPLETE! 🎉")
        return
    
    print(f"NEXT {min(limit, len(results))} COMMANDS TO REFACTOR:")
    print("-" * 60)
    
    for i, (name, category, dual_syntax, camel_case, tests, docs, source_file) in enumerate(results, 1):
        status_icons = []
        if dual_syntax:
            status_icons.append("✅ Dual")
        else:
            status_icons.append("❌ Dual")
            
        if camel_case:
            status_icons.append("✅ Camel")
        else:
            status_icons.append("❌ Camel")
            
        if tests:
            status_icons.append("✅ Tests")
        else:
            status_icons.append("❌ Tests")
            
        if docs:
            status_icons.append("✅ Docs")
        else:
            status_icons.append("❌ Docs")
        
        print(f"{i:2}. {name}")
        print(f"    Category: {category}")
        print(f"    Source: {source_file}")
        print(f"    Status: {' | '.join(status_icons)}")
        print()

def print_category_breakdown(conn):
    """Print breakdown by category."""
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT category, 
               COUNT(*) as total,
               SUM(CASE WHEN is_complete = 1 THEN 1 ELSE 0 END) as complete
        FROM commands 
        GROUP BY category
        ORDER BY total DESC
    ''')
    
    results = cursor.fetchall()
    
    print("PROGRESS BY CATEGORY:")
    print("-" * 40)
    
    for category, total, complete in results:
        percentage = complete/total*100 if total > 0 else 0
        print(f"{category:20} {complete:3}/{total:3} ({percentage:5.1f}%)")
    
    print()

def print_recent_activity(conn, limit=5):
    """Print recent refactoring activity."""
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT command_name, action, details, timestamp
        FROM refactoring_log
        WHERE action != 'scanned'
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    
    if results:
        print("RECENT ACTIVITY:")
        print("-" * 30)
        
        for command_name, action, details, timestamp in results:
            print(f"{timestamp}: {command_name} - {action}")
            if details:
                print(f"    {details}")
        print()

def search_commands(conn, pattern):
    """Search for commands matching a pattern."""
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, category, has_dual_syntax, has_camel_case_alias,
               has_tests, has_documentation, is_complete
        FROM commands 
        WHERE name LIKE ?
        ORDER BY name
    ''', (f'%{pattern}%',))
    
    results = cursor.fetchall()
    
    if not results:
        print(f"No commands found matching '{pattern}'")
        return
    
    print(f"COMMANDS MATCHING '{pattern}':")
    print("-" * 50)
    
    for name, category, dual_syntax, camel_case, tests, docs, complete in results:
        status = "✅ COMPLETE" if complete else "❌ INCOMPLETE"
        print(f"{name:30} {status}")
        print(f"  Category: {category}")
        
        if not complete:
            missing = []
            if not dual_syntax: missing.append("dual syntax")
            if not camel_case: missing.append("camelCase alias")
            if not tests: missing.append("tests")
            if not docs: missing.append("documentation")
            
            if missing:
                print(f"  Missing: {', '.join(missing)}")
        print()

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage: python3 query_next_commands.py [options]")
        print("Options:")
        print("  --help              Show this help")
        print("  --summary           Show summary only")
        print("  --next [N]          Show next N commands (default: 10)")
        print("  --category          Show category breakdown")
        print("  --recent [N]        Show recent N activities (default: 5)")
        print("  --search PATTERN    Search for commands matching pattern")
        print("  --all               Show all incomplete commands")
        print()
        print("Examples:")
        print("  python3 query_next_commands.py")
        print("  python3 query_next_commands.py --next 5")
        print("  python3 query_next_commands.py --search tensor")
        print("  python3 query_next_commands.py --category")
        return
    
    db_path = './refactoring_progress.db'
    conn = connect_db(db_path)
    
    # Parse arguments
    args = sys.argv[1:]
    
    if not args or '--summary' in args:
        print_summary(conn)
    
    if not args or '--next' in args:
        limit = 10
        if '--next' in args:
            try:
                idx = args.index('--next')
                if idx + 1 < len(args) and args[idx + 1].isdigit():
                    limit = int(args[idx + 1])
            except (ValueError, IndexError):
                pass
        print_next_commands(conn, limit)
    
    if '--all' in args:
        print_next_commands(conn, 1000)  # Large number to get all
    
    if '--category' in args:
        print_category_breakdown(conn)
    
    if '--recent' in args:
        limit = 5
        try:
            idx = args.index('--recent')
            if idx + 1 < len(args) and args[idx + 1].isdigit():
                limit = int(args[idx + 1])
        except (ValueError, IndexError):
            pass
        print_recent_activity(conn, limit)
    
    if '--search' in args:
        try:
            idx = args.index('--search')
            if idx + 1 < len(args):
                pattern = args[idx + 1]
                search_commands(conn, pattern)
            else:
                print("Error: --search requires a pattern")
        except ValueError:
            print("Error: --search requires a pattern")
    
    conn.close()

if __name__ == '__main__':
    main() 