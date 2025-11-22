"""
Chatbot module with conversation history management.
Q5: Construction de chatbot
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from .config_loader import ConfigLoader
from .qa_system import QASystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationHistory:
    """Manages conversation history for the chatbot."""
    
    def __init__(self, max_history: int = 5):
        """
        Initialize conversation history.
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_turn(self, question: str, answer: str):
        """
        Add a conversation turn to history.
        
        Args:
            question: User's question
            answer: Bot's answer
        """
        self.history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only the last max_history turns
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history_text(self) -> str:
        """
        Get formatted history text for inclusion in prompts.
        
        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return ""
        
        history_parts = []
        for i, turn in enumerate(self.history):
            history_parts.append(f"User: {turn['question']}")
            history_parts.append(f"Assistant: {turn['answer']}")
        
        return "\n".join(history_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
    
    def get_all(self) -> List[Dict[str, str]]:
        """
        Get all conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.history


class Chatbot:
    """
    Conversational chatbot built on top of the QA system.
    Maintains conversation history and includes it in prompts for context.
    """
    
    def __init__(self, config: ConfigLoader, qa_system: QASystem = None):
        """
        Initialize the Chatbot.
        
        Args:
            config: ConfigLoader instance with system configuration
            qa_system: Optional QASystem instance
        """
        self.config = config
        self.qa_system = qa_system if qa_system else QASystem(config)
        
        max_history = self.config.get('chatbot.max_history', 5)
        self.conversation = ConversationHistory(max_history)
        
        self.system_prompt = self.config.get(
            'chatbot.system_prompt',
            "You are a helpful assistant that answers questions based on the provided context."
        )
        
        self.prompt_template = self._create_chatbot_prompt_template()
        
        logger.info("Chatbot initialized")

    def _trim_history(self, history: str) -> str:
        """Remove the oldest turn from the history string."""
        if not history or history.strip().lower() == "no previous conversation.":
            return "No previous conversation."

        lines = [line for line in history.strip().splitlines() if line.strip()]
        if len(lines) <= 4:
            return "No previous conversation."

        trimmed_lines = lines[2:]
        return "\n".join(trimmed_lines) if trimmed_lines else "No previous conversation."

    def _prepare_prompt(
        self,
        message: str,
        context_parts: List[str],
        history_text: str
    ) -> tuple[str, str, str]:
        """Construct a prompt that respects the token budget."""
        tokenizer = self.qa_system.tokenizer
        max_prompt_tokens = int(self.config.get('chatbot.max_prompt_tokens', 0) or 0)
        default_history = "No previous conversation."
        history = history_text if history_text else default_history
        parts = [part for part in context_parts if part.strip()]
        default_context = "No relevant context was retrieved."

        if not tokenizer or max_prompt_tokens <= 0:
            context = "\n\n".join(parts) if parts else default_context
            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                conversation_history=history,
                context=context,
                question=message
            )
            return prompt, context, history

        while True:
            context = "\n\n".join(parts) if parts else default_context
            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                conversation_history=history,
                context=context,
                question=message
            )

            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_count <= max_prompt_tokens:
                return prompt, context, history

            modified = False

            if parts:
                trimmed = self.qa_system._trim_block(parts[-1])
                if trimmed:
                    parts[-1] = trimmed
                else:
                    parts.pop()
                modified = True

            if not modified and history != default_history:
                history = self._trim_history(history)
                modified = True

            if not modified:
                return prompt, context, history
    
    def _create_chatbot_prompt_template(self) -> PromptTemplate:
        """
        Create a prompt template that includes conversation history.
        
        Returns:
            PromptTemplate instance for chatbot
        """
        template = """You are a helpful AI assistant engaged in a conversation with a user.
You answer questions based on the provided context from documents.

System Instructions:
{system_prompt}

Conversation History:
{conversation_history}

Current Context from Documents:
{context}

Current Question: {question}

Instructions for your response:
1. Consider the conversation history to understand the context and any references to previous exchanges
2. Use the document context to provide accurate, grounded answers
3. If the user refers to something from earlier in the conversation, acknowledge it
4. If the answer requires information from previous turns, incorporate that naturally
5. If you cannot answer based on the context, say so clearly
6. Keep your answers conversational but informative, in English, and avoid repeating raw matrix notation such as [A, B]T unless it is central to the explanation
7. Limit your response to at most four sentences

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["system_prompt", "conversation_history", "context", "question"]
        )
        
        return prompt
    
    def chat(
        self, 
        message: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: User's message/question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary containing response, sources, and conversation info
        """
        logger.info(f"Processing chat message: {message}")
        
        try:
            # Get relevant context from vector store
            retriever = self.qa_system.retriever
            raw_results = retriever.search(message)
            retrieved_docs: List[Document] = []
            for item in raw_results:
                doc = item[0] if isinstance(item, tuple) else item
                if isinstance(doc, Document):
                    retrieved_docs.append(doc)

            max_context_docs = self.config.get('chatbot.max_context_docs', 3)
            max_document_length = self.config.get('chatbot.max_document_length', 400)
            limited_docs = retrieved_docs[:max_context_docs]
            
            # Format context
            context_parts = self.qa_system._build_context_parts(
                limited_docs,
                max_document_length,
                message
            )
            
            # Get conversation history
            history_text = self.conversation.get_history_text()
            prompt, context, history_text = self._prepare_prompt(message, context_parts, history_text)
            
            # Generate answer
            raw_answer = self.qa_system.llm.invoke(prompt)
            if isinstance(raw_answer, str):
                answer = raw_answer
            elif isinstance(raw_answer, list) and raw_answer:
                first = raw_answer[0]
                if isinstance(first, dict) and 'generated_text' in first:
                    answer = first['generated_text']
                else:
                    answer = str(first)
            elif isinstance(raw_answer, dict) and 'generated_text' in raw_answer:
                answer = raw_answer['generated_text']
            else:
                answer = str(raw_answer)

            answer = self.qa_system.normalize_answer(answer)
            
            # Add to conversation history
            self.conversation.add_turn(message, answer)
            
            # Prepare response
            response = {
                "message": message,
                "response": answer,
                "turn_number": len(self.conversation.history),
                "history_length": len(self.conversation.history)
            }
            
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in limited_docs
                ]
                response["num_sources"] = len(limited_docs)
            
            logger.info("Chat response generated successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "message": message,
                "response": f"I encountered an error: {str(e)}",
                "turn_number": len(self.conversation.history),
                "error": True
            }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the full conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation.get_all()
    
    def print_response(self, response: Dict[str, Any]):
        """
        Print chatbot response in a formatted way.
        
        Args:
            response: Response dictionary from chat()
        """
        print(f"\n{'='*80}")
        print(f"Turn #{response.get('turn_number', '?')}")
        print(f"{'='*80}\n")
        print(f"You: {response['message']}\n")
        print(f"Bot: {response['response']}\n")
        
        if "sources" in response and response["sources"]:
            print(f"{'='*80}")
            print(f"Sources: {response['num_sources']} documents used")
            print(f"{'='*80}\n")
    
    def run_interactive_session(self):
        """
        Run an interactive chat session in the console.
        Type 'quit', 'exit', or 'bye' to end the session.
        Type 'history' to see conversation history.
        Type 'reset' to clear conversation history.
        """
        print(f"\n{'='*80}")
        print("Interactive Chatbot Session")
        print(f"{'='*80}\n")
        print("Commands:")
        print("  - Type your question to chat")
        print("  - 'history' to view conversation history")
        print("  - 'reset' to clear conversation history")
        print("  - 'quit', 'exit', or 'bye' to end session\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Thanks for chatting.")
                    break
                
                if user_input.lower() == 'history':
                    self._print_history()
                    continue
                
                if user_input.lower() == 'reset':
                    self.reset_conversation()
                    print("\n✓ Conversation history cleared.\n")
                    continue
                
                # Process the message
                response = self.chat(user_input, return_sources=False)
                print(f"\nBot: {response['response']}\n")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    def _print_history(self):
        """Print conversation history."""
        history = self.get_conversation_history()
        
        if not history:
            print("\nNo conversation history yet.\n")
            return
        
        print(f"\n{'='*80}")
        print(f"Conversation History ({len(history)} turns)")
        print(f"{'='*80}\n")
        
        for i, turn in enumerate(history):
            print(f"[Turn {i+1}]")
            print(f"You: {turn['question']}")
            print(f"Bot: {turn['answer']}")
            print(f"Time: {turn['timestamp']}\n")
    
    def save_conversation(self, filepath: str):
        """
        Save conversation history to a file.
        
        Args:
            filepath: Path to save the conversation
        """
        import json
        
        history = self.get_conversation_history()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation saved to {filepath}")
        print(f"\n✓ Conversation saved to {filepath}\n")
