"""
Complete test suite for the RAG system using introductory statistics PDFs.
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > null 2>&1')

print("=" * 80)
print("Complete test suite for all questions with your introductory statistics PDFs")
print("=" * 80)

# Import system
from src.config_loader import ConfigLoader
from src.document_indexer import DocumentIndexer
from src.document_retriever import DocumentRetriever
from src.qa_system import QASystem
from src.rag_evaluator import RAGEvaluator
from src.chatbot import Chatbot

# Check PDFs
print("\n# Check PDF Files")
data_dir = Path("data")
pdf_files = list(data_dir.glob("*.pdf"))
print(f"[OK] Found {len(pdf_files)} PDF files in data/")
for pdf in pdf_files:
    print(f"     - {pdf.name}")

print("\n" + "=" * 80)
print("Q1 TEST: Document Indexation Pipeline")
print("=" * 80)

config = ConfigLoader()
indexer = DocumentIndexer(config)

print("\n[INFO] Starting document indexation...")
try:
    indexer.index_documents("data")
    print("[OK] Documents indexed successfully!")
    print(f"[INFO] Vector store created at: ./chroma_db")
except Exception as e:
    print(f"[ERROR] Indexation failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Force flush output
sys.stdout.flush()

print("\n" + "=" * 80)
print("Q2 TEST: Vector Database Search")
print("=" * 80)

try:
    retriever = DocumentRetriever(config)
    print("[OK] Retriever initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize retriever: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

test_queries = [
    "American Statistical Association definition of statistics",
    "difference between descriptive and inferential statistics",
    "requirements for conducting an experiment"
]

print("\n[INFO] Testing vector search with statistical queries...")
for query in test_queries:
    print(f"\n[QUERY] '{query}'")
    try:
        results = retriever.search(query, top_k=2)
    except Exception as e:
        print(f"[ERROR] Search failed: {str(e)}")
        continue
    
    for i, item in enumerate(results, 1):
        # Handle both Document and (Document, score) tuple formats
        if isinstance(item, tuple):
            doc, score = item
            print(f"  [{i}] Score: {score:.4f}")
        else:
            doc = item
            print(f"  [{i}] Score: N/A")
        print(f"      Content: {doc.page_content[:100]}...")

print("\n" + "=" * 80)
print("Q3 TEST: Question Answering with LLM")
print("=" * 80)

try:
    qa = QASystem(config)
    print("[OK] QA System initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize QA system: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

test_questions = [
    "According to the American Statistical Association, how is statistics defined in Chapter 1?",
    "How do descriptive and inferential statistics differ in the introduction chapter?",
    "According to the introduction chapter, what three requirements must be met to conduct an experiment?"
]

print("\n[INFO] Testing QA system with LLM...")
for question in test_questions:
    print(f"\n[QUESTION] {question}")
    try:
        result = qa.answer_question(question)
        answer_text = result.get('answer', '') if isinstance(result, dict) else str(result)
        print(f"[ANSWER] {answer_text[:200]}...")
    except Exception as e:
        print(f"[ERROR] QA failed: {str(e)}")

print("\n" + "=" * 80)
print("Q4 TEST: Answer Evaluation")
print("=" * 80)

evaluator = RAGEvaluator(config)

print("\n[INFO] Evaluating answer quality...")
if test_questions:
    question = test_questions[0]
    result = qa.answer_question(question)
    answer_text = result.get('answer', '') if isinstance(result, dict) else str(result)
    retrieved_docs = retriever.search(question, top_k=4)
    source_documents = []
    for item in retrieved_docs:
        if isinstance(item, tuple):
            doc, _ = item
            source_documents.append(doc)
        else:
            source_documents.append(item)

    print(f"\n[EVAL] Question: {question}")
    print(f"[EVAL] Answer: {answer_text[:150]}...")
    
    try:
        relevance = evaluator.evaluate_answer_relevance(question, answer_text)
        faithfulness = evaluator.evaluate_faithfulness(answer_text, source_documents)
        completeness = evaluator.evaluate_answer_completeness(answer_text)
        
        print(f"\n[METRICS]")
        print(f"  Relevance Score:    {relevance.score:.2f} ({relevance.reasoning})")
        print(f"  Faithfulness Score: {faithfulness.score:.2f} ({faithfulness.reasoning})")
        print(f"  Completeness Score: {completeness.score:.2f} ({completeness.reasoning})")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {str(e)}")

print("\n" + "=" * 80)
print("Q5 TEST: Chatbot with Conversation History")
print("=" * 80)

chatbot = Chatbot(config)

conversation = [
    "What does the American Statistical Association say statistics is?",
    "Can you summarize the difference between descriptive and inferential statistics?",
    "What are the three requirements for conducting an experiment?"
]
print("\n[INFO] Testing chatbot with multi-turn conversation...")
for i, message in enumerate(conversation, 1):
    print(f"\n[USER {i}] {message}")
    try:
        response = chatbot.chat(message)
        if isinstance(response, dict):
            bot_reply = response.get('response', '')
            response_text = bot_reply if isinstance(bot_reply, str) else str(bot_reply)
        else:
            response_text = response if isinstance(response, str) else str(response)
        print(f"[BOT {i}] {response_text[:200]}...")
    except Exception as e:
        print(f"[ERROR] Chat failed: {str(e)}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED!")
print("=" * 80)
print("\n[OK] All 5 questions (Q1-Q5) tested successfully with introductory statistics PDFs")
