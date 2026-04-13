import sys
import asyncio
from main import main as interactive_chat
from batch_main import async_batch_process, clear_answers

def menu():
    while True:
        print("\n" + "="*40)
        print("   🚀 DocLens Master Control Panel")
        print("="*40)
        print("1. 💬 Interactive Chat Mode")
        print("2. 📑 Batch Process questions.txt (Async/Concurrency 2)")
        print("3. 🗑️  Clear answers.txt")
        print("4. ❌ Exit")
        print("="*40)
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            try:
                interactive_chat()
            except KeyboardInterrupt:
                print("\n\n↩️  Returning to Master Menu...")
        
        elif choice == "2":
            try:
                asyncio.run(async_batch_process())
            except KeyboardInterrupt:
                print("\n\n↩️  Batch process interrupted. Returning to menu...")
        
        elif choice == "3":
            clear_answers()
        
        elif choice == "4" or choice.lower() == "exit":
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
