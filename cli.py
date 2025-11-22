"""Command-line interface for the RAG project."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional

from langchain_core.documents import Document

from src.config_loader import ConfigLoader
from src.document_indexer import DocumentIndexer
from src.document_retriever import DocumentRetriever
from src.qa_system import QASystem
from src.rag_evaluator import RAGEvaluator
from src.chatbot import Chatbot


def configure_logging(verbose: bool) -> None:
    """Configure application logging verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def load_config(config_path: str) -> ConfigLoader:
    """Load the YAML configuration file."""
    resolved = Path(config_path).resolve()
    return ConfigLoader(str(resolved))


def docs_from_results(results: Iterable[Any]) -> List[Document]:
    """Extract Document objects from retriever outputs."""
    documents: List[Document] = []
    for item in results:
        if isinstance(item, tuple):
            doc = item[0]
        else:
            doc = item
        if isinstance(doc, Document):
            documents.append(doc)
    return documents


def handle_index(args: argparse.Namespace, config: ConfigLoader) -> None:
    indexer = DocumentIndexer(config)
    indexer.index_documents(args.data)
    print("\n✓ Document indexation completed.")


def handle_retrieve(args: argparse.Namespace, config: ConfigLoader) -> None:
    retriever = DocumentRetriever(config)
    results = retriever.search(args.query, top_k=args.top_k, return_scores=True)
    retriever.print_results(results, max_content_length=args.max_content)

    if args.json:
        formatted = retriever.format_results(results)
        print(json.dumps(formatted, indent=2, ensure_ascii=False))


def handle_ask(args: argparse.Namespace, config: ConfigLoader) -> None:
    qa_system = QASystem(config)
    result = qa_system.answer_question(args.question, return_sources=args.sources)
    qa_system.print_answer(result)

    if args.json:
        serializable = {
            "question": result["question"],
            "answer": result["answer"],
        }
        if args.sources:
            serializable["sources"] = result.get("sources", [])
        print(json.dumps(serializable, indent=2, ensure_ascii=False))


def handle_evaluate(args: argparse.Namespace, config: ConfigLoader) -> None:
    qa_system = QASystem(config)
    evaluator = RAGEvaluator(config, qa_system=qa_system)

    if args.answer:
        retriever = qa_system.retriever
        results = retriever.search(args.question, top_k=args.top_k or retriever.top_k, return_scores=False)
        documents = docs_from_results(results)
        evaluation = evaluator.evaluate_response(args.question, args.answer, documents)
    else:
        qa_result = qa_system.answer_question(args.question, return_sources=True)
        evaluation = evaluator.evaluate_qa_result(qa_result)

    evaluator.print_evaluation(evaluation)

    if args.json:
        print(json.dumps(evaluation, indent=2, ensure_ascii=False))


def handle_chat(args: argparse.Namespace, config: ConfigLoader) -> None:
    chatbot = Chatbot(config)
    if args.message:
        response = chatbot.chat(args.message, return_sources=args.sources)
        chatbot.print_response(response)
        if args.json:
            print(json.dumps(response, indent=2, ensure_ascii=False))
        return

    chatbot.run_interactive_session()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="CLI for the RAG pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index documents into the vector store")
    index_parser.add_argument("--data", help="Directory with source documents (defaults to config value)")
    index_parser.set_defaults(func=handle_index)

    retrieve_parser = subparsers.add_parser("retrieve", help="Search the vector store")
    retrieve_parser.add_argument("query", help="Natural language query to run")
    retrieve_parser.add_argument("--top-k", type=int, help="Number of documents to return")
    retrieve_parser.add_argument("--max-content", type=int, default=200, help="Characters to display per result")
    retrieve_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    retrieve_parser.set_defaults(func=handle_retrieve)

    ask_parser = subparsers.add_parser("ask", help="Pose a question to the QA system")
    ask_parser.add_argument("question", help="User question")
    ask_parser.add_argument("--sources", action="store_true", help="Include source snippets in the response")
    ask_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    ask_parser.set_defaults(func=handle_ask)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate an answer with RAG metrics")
    evaluate_parser.add_argument("question", help="Question to evaluate")
    evaluate_parser.add_argument("--answer", help="Optional precomputed answer to score")
    evaluate_parser.add_argument("--top-k", type=int, help="Documents to retrieve when scoring a custom answer")
    evaluate_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    evaluate_parser.set_defaults(func=handle_evaluate)

    chat_parser = subparsers.add_parser("chat", help="Chat with the system")
    chat_parser.add_argument("--message", help="Single-turn message instead of interactive mode")
    chat_parser.add_argument("--sources", action="store_true", help="Include supporting sources in responses")
    chat_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    chat_parser.set_defaults(func=handle_chat)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    config = load_config(args.config)

    args.func(args, config)


if __name__ == "__main__":
    main()
