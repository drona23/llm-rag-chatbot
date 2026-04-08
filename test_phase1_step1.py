"""
PHASE 1 STEP 1 TEST: generate_response() Function

This script tests if the generate_response() function works correctly.
We're testing with student loan documents we created.

Test Flow:
1. Initialize LLMChat
2. Load student loan documents
3. Test with different questions
4. Check if Claude uses documents correctly
"""

from src.llm import LLMChat
import os
import pathlib

BASE = pathlib.Path(__file__).parent

print("=" * 70)
print("PHASE 1 STEP 1 TEST: RAG Response Generation")
print("=" * 70)

# Initialize Claude
print("\n[STEP 1] Initializing LLMChat...")
llm = LLMChat()
print("✓ LLMChat initialized")

# Load student loan documents from our data files
print("\n[STEP 2] Loading student loan documents...")

# Read the documents we created
docs_to_load = {
    "loan_types.txt": BASE / "data/student_loans/loan_types.txt",
    "interest_rates.txt": BASE / "data/student_loans/interest_rates.txt",
}

documents = []
for name, path in docs_to_load.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
            documents.append(content)
            print(f"✓ Loaded {name} ({len(content)} characters)")
    else:
        print(f"✗ File not found: {path}")

print(f"\nTotal documents loaded: {len(documents)}")

# Test questions
test_questions = [
    "What are the main types of federal student loans?",
    "What's the current interest rate for direct unsubsidized loans?",
    "What's the difference between subsidized and unsubsidized loans?",
]

print("\n" + "=" * 70)
print("RUNNING TESTS")
print("=" * 70)

# Test each question
for i, question in enumerate(test_questions, 1):
    print(f"\n[TEST {i}]")
    print(f"Question: {question}")
    print(f"Documents provided: {len(documents)}")
    print("-" * 70)

    try:
        # Generate response using the documents
        response = llm.generate_response(
            user_message=question,
            context_docs=documents
        )

        print(f"Response:\n{response}")
        print("-" * 70)

        # Quick checks
        print("Quality Checks:")
        if len(response) > 50:
            print("✓ Response length reasonable")
        else:
            print("⚠ Response might be too short")

        if "Document" in response or "documents" in response.lower() or "based on" in response.lower():
            print("✓ Claude cited sources")
        else:
            print("⚠ Claude might not have cited sources")

        if "loan" in response.lower():
            print("✓ Response mentions loans (on-topic)")
        else:
            print("⚠ Response might be off-topic")

    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print("\nNotes:")
print("- If all tests show ✓, generate_response() is working!")
print("- If there are errors, check:")
print("  • Your ANTHROPIC_API_KEY environment variable is set")
print("  • Internet connection is working")
print("  • Claude API account has credits")
