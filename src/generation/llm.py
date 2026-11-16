"""
Provider-agnostic LLM interface and Ollama implementation for CodeCompass.

Enables swappable LLM backends:
  - Default: local Ollama with `qwen2.5-coder:3b`
  - Future: Gemini, Groq, Anthropic, OpenAI via the same LLMProvider interface

Provides explicit error handling if Ollama is stopped or the model is not pulled.
"""

from abc import ABC, abstractmethod
from typing import Iterator

from src.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
)


# ── Base Abstract Interface ──────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base class for all language model providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a complete text response for a given prompt.

        Args:
            prompt: User/assistant conversation prompt.
            system_prompt: System instruction directing model behavior.

        Returns:
            The model's response string.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        """
        Stream tokens from the model chunk by chunk (for real-time UI display).

        Args:
            prompt: User/assistant conversation prompt.
            system_prompt: System instruction directing model behavior.

        Yields:
            Text token deltas as they are generated.
        """
        pass

    @abstractmethod
    def check_health(self) -> tuple[bool, str]:
        """
        Check if the LLM backend is available and the model is loaded.

        Returns:
            (True, "OK") if healthy, or (False, "Error explanation") if not.
        """
        pass


# ── Ollama Provider (Local) ──────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Local Ollama LLM provider.

    Connects to an Ollama instance (default: http://localhost:11434) and runs
    small code-specialized models like `qwen2.5-coder:3b`.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        num_ctx: int = LLM_NUM_CTX,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx

        import ollama
        self._client = ollama.Client(host=self.base_url)

    def check_health(self) -> tuple[bool, str]:
        """
        Verify that the Ollama daemon is reachable and the model is downloaded.
        """
        try:
            models_response = self._client.list()
        except Exception as exc:
            return (
                False,
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. (Details: {exc})",
            )

        # Extract available model tags
        available_models: list[str] = []
        try:
            # Handle both dict responses and object responses from ollama-python
            models_list = getattr(models_response, "models", None)
            if models_list is None and isinstance(models_response, dict):
                models_list = models_response.get("models", [])

            for m in models_list or []:
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if not name and isinstance(m, dict):
                    name = m.get("model") or m.get("name")
                if name:
                    available_models.append(name.lower())
        except Exception:
            pass

        # Check if the requested model (e.g. 'qwen2.5-coder:3b') is pulled
        target = self.model.lower()
        model_matched = any(
            target == m or m.startswith(f"{target}:") or target.startswith(f"{m}:")
            for m in available_models
        )

        if not model_matched:
            return (
                False,
                f"Model '{self.model}' is not pulled in Ollama. "
                f"Available models: {available_models}. "
                f"Please run: `ollama pull {self.model}` in your terminal.",
            )

        return (True, f"Ollama is ready (model: {self.model})")

    def _prepare_messages(self, prompt: str, system_prompt: str) -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response synchronously."""
        healthy, message = self.check_health()
        if not healthy:
            raise RuntimeError(message)

        messages = self._prepare_messages(prompt, system_prompt)

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                },
            )
            # Response may be object or dict
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content
            return response["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        """Stream response tokens as they arrive."""
        healthy, message = self.check_health()
        if not healthy:
            raise RuntimeError(message)

        messages = self._prepare_messages(prompt, system_prompt)

        try:
            stream_gen = self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                },
            )
            for chunk in stream_gen:
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    yield chunk.message.content
                elif isinstance(chunk, dict) and "message" in chunk:
                    yield chunk["message"].get("content", "")
        except Exception as exc:
            raise RuntimeError(f"Ollama streaming failed: {exc}") from exc


# ── Provider Factory ─────────────────────────────────────────────────────────

def get_llm_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
) -> LLMProvider:
    """
    Factory to instantiate the configured LLM provider.

    Args:
        provider_name: 'ollama' (default) or future providers ('gemini', 'groq').
        model_name: Model identifier (defaults to LLM_MODEL from config).

    Returns:
        An instance implementing the LLMProvider interface.
    """
    provider = (provider_name or LLM_PROVIDER).lower()
    model = model_name or LLM_MODEL

    if provider == "ollama":
        return OllamaProvider(model=model)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: ['ollama']"
        )


if __name__ == "__main__":
    # Quick health check test from CLI
    llm = get_llm_provider()
    healthy, msg = llm.check_health()
    print(f"Health Status: {'✅' if healthy else '❌'} {msg}")

    if healthy:
        print("\nTesting test generation with Ollama...")
        test_out = llm.generate("Say 'CodeCompass is ready' in 5 words or less.")
        print(f"Response: {test_out.strip()}")

