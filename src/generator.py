"""
LLM Generator Module

Generates responses using Ollama and Llama 3.2.
"""

import ollama
from typing import List, Dict, Optional


class LLMGenerator:
    """Class for generating responses using Ollama LLM."""
    
    def __init__(self, model: str = 'llama3.2'):
        """
        Initialize the LLM generator.
        
        Args:
            model: Name of the Ollama model to use
        """
        self.model = model
        self._check_model_availability()
    
    def _check_model_availability(self) -> None:
        """Check if the specified model is available in Ollama."""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models['models']]
            
            if not any(self.model in name for name in model_names):
                print(f"Warning: Model '{self.model}' not found in Ollama.")
                print(f"Please run: ollama pull {self.model}")
        except Exception as e:
            print(f"Warning: Could not check model availability: {e}")
    
    def generate(self, query: str, context: List[Dict], 
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate a response given a query and context.
        
        Args:
            query: User query
            context: List of relevant context chunks
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            Generated response string
        """
        # Build prompt with context
        context_text = "\n\n".join([chunk['text'] for chunk in context])
        
        prompt = f"""You are a helpful assistant answering questions based on the provided context. Use only the information from the context to answer the question. If the answer cannot be found in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Generate response using Ollama
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature
                }
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_streaming(self, query: str, context: List[Dict],
                          max_tokens: int = 512, temperature: float = 0.7):
        """
        Generate a response with streaming output.
        
        Args:
            query: User query
            context: List of relevant context chunks
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Yields:
            Chunks of generated text
        """
        # Build prompt with context
        context_text = "\n\n".join([chunk['text'] for chunk in context])
        
        prompt = f"""You are a helpful assistant answering questions based on the provided context. Use only the information from the context to answer the question. If the answer cannot be found in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Stream response using Ollama
        try:
            stream = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature
                },
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def chat(self, messages: List[Dict], max_tokens: int = 512,
             temperature: float = 0.7) -> str:
        """
        Chat with the LLM using a conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            Generated response string
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature
                }
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error generating response: {str(e)}"


def generate_response(query: str, context: List[Dict], 
                     model: str = 'llama3.2') -> str:
    """
    Convenience function to generate a response.
    
    Args:
        query: User query
        context: List of context chunks
        model: Model name to use
    
    Returns:
        Generated response
    """
    generator = LLMGenerator(model=model)
    return generator.generate(query=query, context=context)
