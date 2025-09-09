#!/usr/bin/env python3
"""
Update command status in the SQLite tracking database.
Use this script to mark commands as completed after refactoring.
"""

import sqlite3
import sys
import os
from datetime import datetime

def connect_db(db_path='./refactoring_progress.db'):
    """Connect to the tracking database."""
    if not os.path.exists(db_path):
        print(f"Error: Database '{db_path}' not found. Run init_tracking_db.py first.")
        sys.exit(1)
    return sqlite3.connect(db_path)

def update_command_status(conn, command_name, **kwargs):
    """Update the status of a command in the database."""
    cursor = conn.cursor()
    
    # Check if command exists
    cursor.execute('SELECT name FROM commands WHERE name = ?', (command_name,))
    if not cursor.fetchone():
        print(f"Error: Command '{command_name}' not found in database")
        return False
    
    # Build update query
    set_clauses = []
    params = []
    
    valid_fields = ['has_dual_syntax', 'has_camel_case_alias', 'has_tests', 'has_documentation']
    
    for field, value in kwargs.items():
        if field in valid_fields:
            set_clauses.append(f"{field} = ?")
            params.append(1 if value else 0)
    
    if not set_clauses:
        print("No valid fields to update")
        return False
    
    # Add updated_at
    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
    
    # Update the command
    query = f"UPDATE commands SET {', '.join(set_clauses)} WHERE name = ?"
    params.append(command_name)
    
    cursor.execute(query, params)
    
    # Check if command is now complete
    cursor.execute('''
        SELECT has_dual_syntax, has_camel_case_alias, has_tests, has_documentation
        FROM commands WHERE name = ?
    ''', (command_name,))
    
    dual, camel, tests, docs = cursor.fetchone()
    is_complete = dual and camel and tests and docs
    
    # Update is_complete flag
    cursor.execute('UPDATE commands SET is_complete = ? WHERE name = ?', 
                   (1 if is_complete else 0, command_name))
    
    # Log the action
    details = f"Updated: {', '.join(f'{k}={v}' for k, v in kwargs.items())}"
    cursor.execute('''
        INSERT INTO refactoring_log (command_name, action, details)
        VALUES (?, ?, ?)
    ''', (command_name, 'updated', details))
    
    conn.commit()
    
    print(f"✅ Updated {command_name}")
    print(f"   Status: Dual={dual}, Camel={camel}, Tests={tests}, Docs={docs}")
    if is_complete:
        print(f"   🎉 Command is now COMPLETE!")
    
    return True

def mark_complete(conn, command_name):
    """Mark a command as completely refactored."""
    return update_command_status(conn, command_name,
                                has_dual_syntax=True,
                                has_camel_case_alias=True,
                                has_tests=True,
                                has_documentation=True)

def rescan_command(conn, command_name):
    """Rescan a specific command's status from the codebase."""
    # Import functions from init script
    sys.path.append(os.path.dirname(__file__))
    from init_tracking_db import check_dual_syntax, check_camel_case_alias, check_tests, check_documentation
    
    cursor = conn.cursor()
    
    # Check if command exists
    cursor.execute('SELECT name FROM commands WHERE name = ?', (command_name,))
    if not cursor.fetchone():
        print(f"Error: Command '{command_name}' not found in database")
        return False
    
    # Rescan status
    src_dir = './src'
    has_dual_syntax = check_dual_syntax(command_name, src_dir)
    has_camel_case = check_camel_case_alias(command_name, src_dir)
    has_tests = check_tests(command_name)
    has_docs = check_documentation(command_name)
    is_complete = has_dual_syntax and has_camel_case and has_tests and has_docs
    
    # Update database
    cursor.execute('''
        UPDATE commands 
        SET has_dual_syntax = ?, has_camel_case_alias = ?, has_tests = ?, 
            has_documentation = ?, is_complete = ?, updated_at = CURRENT_TIMESTAMP
        WHERE name = ?
    ''', (has_dual_syntax, has_camel_case, has_tests, has_docs, is_complete, command_name))
    
    # Log the rescan
    cursor.execute('''
        INSERT INTO refactoring_log (command_name, action, details)
        VALUES (?, ?, ?)
    ''', (command_name, 'rescanned', f'Status: dual_syntax={has_dual_syntax}, camel={has_camel_case}, tests={has_tests}, docs={has_docs}'))
    
    conn.commit()
    
    print(f"✅ Rescanned {command_name}")
    print(f"   Status: Dual={has_dual_syntax}, Camel={has_camel_case}, Tests={has_tests}, Docs={has_docs}")
    if is_complete:
        print(f"   🎉 Command is COMPLETE!")
    
    return True

def main():
    """Main function."""
    if len(sys.argv) < 2 or '--help' in sys.argv:
        print("Usage: python3 update_command_status.py <command> [options]")
        print()
        print("Commands:")
        print("  mark-complete COMMAND    Mark command as completely refactored")
        print("  update COMMAND OPTIONS   Update specific aspects of a command")
        print("  rescan COMMAND           Rescan command status from codebase")
        print("  rescan-all               Rescan all commands")
        print()
        print("Update options:")
        print("  --dual-syntax [true|false]       Has dual syntax support")
        print("  --camel-case [true|false]        Has camelCase alias")
        print("  --tests [true|false]             Has test file")
        print("  --docs [true|false]              Has documentation")
        print()
        print("Examples:")
        print("  python3 update_command_status.py mark-complete torch::linear")
        print("  python3 update_command_status.py update torch::conv2d --dual-syntax true --tests true")
        print("  python3 update_command_status.py rescan torch::tensor_add")
        print("  python3 update_command_status.py rescan-all")
        return
    
    db_path = './refactoring_progress.db'
    conn = connect_db(db_path)
    
    command = sys.argv[1]
    
    if command == 'mark-complete':
        if len(sys.argv) < 3:
            print("Error: mark-complete requires a command name")
            sys.exit(1)
        
        command_name = sys.argv[2]
        mark_complete(conn, command_name)
    
    elif command == 'update':
        if len(sys.argv) < 3:
            print("Error: update requires a command name")
            sys.exit(1)
        
        command_name = sys.argv[2]
        
        # Parse update options
        kwargs = {}
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == '--dual-syntax' and i + 1 < len(args):
                kwargs['has_dual_syntax'] = args[i + 1].lower() == 'true'
                i += 2
            elif args[i] == '--camel-case' and i + 1 < len(args):
                kwargs['has_camel_case_alias'] = args[i + 1].lower() == 'true'
                i += 2
            elif args[i] == '--tests' and i + 1 < len(args):
                kwargs['has_tests'] = args[i + 1].lower() == 'true'
                i += 2
            elif args[i] == '--docs' and i + 1 < len(args):
                kwargs['has_documentation'] = args[i + 1].lower() == 'true'
                i += 2
            else:
                print(f"Unknown option: {args[i]}")
                i += 1
        
        if not kwargs:
            print("No update options provided")
            sys.exit(1)
        
        update_command_status(conn, command_name, **kwargs)
    
    elif command == 'rescan':
        if len(sys.argv) < 3:
            print("Error: rescan requires a command name")
            sys.exit(1)
        
        command_name = sys.argv[2]
        rescan_command(conn, command_name)
    
    elif command == 'rescan-all':
        print("Rescanning all commands...")
        
        # Import and run the populate function
        sys.path.append(os.path.dirname(__file__))
        from init_tracking_db import populate_database
        
        src_dir = './src'
        populate_database(conn, src_dir)
        print("✅ All commands rescanned")
    
    else:
        print(f"Unknown command: {command}")
        print("Use --help for usage information")
        sys.exit(1)
    
    conn.close()

if __name__ == '__main__':
    main() 