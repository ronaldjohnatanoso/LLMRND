#!/usr/bin/env python3
"""Reset CogMemory LanceDB database.

Usage:
    python reset_db.py              # Reset default database
    python reset_db.py --path ./custom/path  # Reset custom database
    python reset_db.py --yes        # Skip confirmation
"""

import argparse
import shutil
from pathlib import Path

from cog_memory.lance_store import LanceStore


def reset_database(db_path: str = "./data/lancedb", force: bool = False) -> None:
    """Reset the LanceDB database.

    Args:
        db_path: Path to the database directory
        force: Skip confirmation prompt
    """
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"✅ Database does not exist at {db_path}")
        return

    # Count nodes before deletion
    try:
        store = LanceStore(db_path=db_path)
        count = store.count_nodes()
        print(f"Found {count} nodes in database")
    except Exception:
        count = "unknown"
        print(f"Could not count nodes (database may be corrupted)")

    # Confirmation
    if not force:
        response = input(f"⚠️  Delete database at {db_path}? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("❌ Cancelled")
            return

    # Delete
    print(f"🗑️  Deleting database at {db_path}...")
    shutil.rmtree(db_path)
    print("✅ Database deleted")

    # Verify
    if not db_path.exists():
        print("✅ Verification: Database successfully removed")
    else:
        print("❌ Warning: Database still exists")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reset CogMemory LanceDB database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Reset with confirmation
  %(prog)s --yes            # Reset without confirmation
  %(prog)s --path ./custom  # Reset custom database location
        """,
    )
    parser.add_argument(
        "--path",
        default="./data/lancedb",
        help="Path to database directory (default: ./data/lancedb)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("🧠 CogMemory Database Reset")
    print("=" * 50)
    print()

    reset_database(db_path=args.path, force=args.yes)


if __name__ == "__main__":
    main()
