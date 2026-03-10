"""
RAGBrain — LLM Generator
Supports Ollama (offline) and Groq (online) providers.
"""

from typing import List, Dict
import os

DEFAULT_MODEL    = "llama3"
DEFAULT_PROVIDER = "ollama"

LISTING_PROMPT = """You are reading a resume. Find the PROJECTS section only.

{context}

---
Question: {question}

EXAMPLE of how projects appear in the document:
  Hybrid Chess Engine | Python | PyTorch | Pygame
  Lead Flow | n8n | Google Sheets API
  Doc Chat | LangChain | Hugging Face

The answer to "name the projects" for the example above would be:
1. Hybrid Chess Engine
2. Lead Flow
3. Doc Chat

Now answer the question using the same format.
Only include project names — nothing else.
Do not include technologies after the "|" symbol.
Do not include bullet points or descriptions.
Do not include experience or internship entries.

Answer:"""

GENERAL_PROMPT = """You are a highly intelligent assistant analyzing a document.

{context}

---
Question: {question}

Instructions:
- Answer based on the document content
- Be insightful, specific and detailed
- For project questions: describe what it does, technologies used, and key achievements
- For recommendation questions: give creative, personalized suggestions based on the person's actual skills
- Do not just rephrase what's written — add insight and context

Answer:"""


def is_listing_question(question: str) -> bool:
    keywords = ["list", "name", "what are", "how many", "mention", "all project",
                "3 project", "the project", "projects in", "projects mention"]
    return any(kw in question.lower() for kw in keywords)


class Generator:
    def __init__(self, model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER):
        self.model    = model
        self.provider = provider
        self._groq_client = None
        print(f"[Generator] provider={provider} model={model}")

    def _get_groq_client(self):
        if self._groq_client is None:
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "GROQ_API_KEY not set.\n"
                        "1. Go to https://console.groq.com\n"
                        "2. Sign up free and create an API key\n"
                        "3. Add to your .env file: GROQ_API_KEY=your_key_here"
                    )
                self._groq_client = Groq(api_key=api_key)
            except ImportError:
                raise RuntimeError("Groq not installed. Run: pip install groq")
        return self._groq_client

    def build_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            context = "No content found."
        else:
            sorted_chunks = sorted(context_chunks, key=lambda x: x.get("chunk_index", 0))
            context = "\n\n".join(c['text'] for c in sorted_chunks)
        template = LISTING_PROMPT if is_listing_question(question) else GENERAL_PROMPT
        return template.format(context=context, question=question)

    def _generate_ollama(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        return response["message"]["content"]

    def _generate_groq(self, prompt: str) -> str:
        client = self._get_groq_client()
        # Map UI model names to Groq model IDs
        groq_model_map = {
            "llama3-70b": "llama-3.3-70b-versatile",
            "llama3-8b":  "llama-3.1-8b-instant",
        }
        groq_model = groq_model_map.get(self.model, "llama3-70b-8192")
        response = self._groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def generate(self, question: str, context_chunks: List[Dict]) -> Dict:
        prompt = self.build_prompt(question, context_chunks)
        ptype  = "LISTING" if is_listing_question(question) else "GENERAL"
        print(f"[Generator] provider={self.provider} model={self.model} prompt={ptype}")

        try:
            if self.provider == "groq":
                # Lazy init groq client
                self._get_groq_client()
                answer = self._generate_groq(prompt)
            else:
                answer = self._generate_ollama(prompt)
        except RuntimeError:
            raise
        except Exception as e:
            if self.provider == "groq":
                raise RuntimeError(f"Groq error: {e}\nCheck your GROQ_API_KEY in .env file.")
            else:
                raise RuntimeError(
                    f"Ollama error. Make sure Ollama is running and '{self.model}' is pulled.\n"
                    f"Run: ollama pull {self.model}\nError: {e}"
                )

        sources = list({c.get("source", "unknown") for c in context_chunks})
        return {
            "answer":             answer,
            "model":              f"{self.provider}/{self.model}",
            "sources":            sources,
            "num_context_chunks": len(context_chunks),
        }