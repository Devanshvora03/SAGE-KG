import os
from pathlib import Path

def replace_in_markdown_files(root_folder: str, dry_run: bool = True) -> None:
    """
    Recursively finds all .md files and performs the following replacements:
    - 'Llama 3.3 70B' → 'GPT-4o-mini'
    - '(via Ollama)' → ''
    """
    root = Path(root_folder).resolve()
    
    if not root.exists():
        print(f"Error: Folder not found → {root}")
        return
    
    if not root.is_dir():
        print(f"Error: Not a directory → {root}")
        return
    
    replacement_count = 0
    files_modified = 0
    files_checked = 0
    
    print(f"Starting in: {root}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'REAL CHANGES WILL BE WRITTEN'}")
    print("-" * 70)
    
    replacements = [
        ("Llama 3.3 70B",      "GPT-4o-mini"),
        ("Llama 3.3 70b",      "GPT-4o-mini"),
        ("llama 3.3 70b",      "gpt-4o-mini"),
        ("Llama3.3 70B",       "GPT-4o-mini"),
        ("(via Ollama)",       ""),
        ("(Via Ollama)",       ""),
        (" via Ollama",        ""),     # sometimes written without parentheses
    ]
    
    for filepath in root.rglob("*.md"):
        files_checked += 1
        try:
            original_content = filepath.read_text(encoding="utf-8")
            content = original_content
            
            # Apply all replacements
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    # Count only the meaningful replacements (not the cleanup ones)
                    if new == "GPT-4o-mini":
                        replacement_count += original_content.count(old)
            
            if content != original_content:
                files_modified += 1
                
                print(f"Found changes in: {filepath.name}")
                print(f"  → {filepath.relative_to(root)}")
                
                if not dry_run:
                    filepath.write_text(content, encoding="utf-8")
                    print("  → File updated")
                else:
                    print("  → Would update (dry run)")
                    
                print()
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print("-" * 70)
    print("Summary:")
    print(f"  Files checked     : {files_checked:,}")
    print(f"  Files modified    : {files_modified:,}")
    print(f"  Model replacements: {replacement_count:,}")
    if dry_run and files_modified > 0:
        print("\nNo files were actually changed (dry run mode).")
        print("Run with dry_run=False to apply the changes.\n")


if __name__ == "__main__":
    # Choose ONE folder at a time
    FOLDER_TO_SEARCH = r"C:\Workspace\SAGE-KG\Ablation"
    # FOLDER_TO_SEARCH = r"C:\Workspace\SAGE-KG\Ablation\Agents"
    # FOLDER_TO_SEARCH = r"C:\Workspace\SAGE-KG"
    
    # Recommended: always test first with dry_run=True
    # replace_in_markdown_files(FOLDER_TO_SEARCH, dry_run=True)

    # After verifying the output looks correct, use this:
    replace_in_markdown_files(FOLDER_TO_SEARCH, dry_run=False)