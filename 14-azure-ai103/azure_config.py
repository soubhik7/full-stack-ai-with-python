"""Centralized Azure credential/project configuration for Chapter 14 (Azure AI-103).

Every script and notebook in this chapter reads its Azure endpoints, keys, project
details, and model deployment names through the single `config` object below instead
of each file duplicating its own `load_dotenv()` + `os.getenv(...)` calls with a
slightly different variable name. Add a value once to the repo-root `.env` and every
script/notebook that needs it finds it the same way.

Usage (from any file inside 14-azure-ai103/, at any folder depth):

    import sys
    from pathlib import Path

    _start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for _parent in [_start, *_start.parents]:
        if (_parent / "azure_config.py").exists():
            sys.path.insert(0, str(_parent))
            break

    from azure_config import config

    project = AIProjectClient(endpoint=config.project_endpoint, credential=DefaultAzureCredential())

See README.md in this folder for the full list of environment variables this reads.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# This file lives at <repo_root>/14-azure-ai103/azure_config.py — the repo-root .env
# is one directory up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


class MissingCredentialError(RuntimeError):
    """Raised when code asks for a credential that isn't set in the repo-root .env."""


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty env var among `names`.

    Multiple names let us accept both the canonical name and older aliases that
    already-existing .env files or course material may use, without breaking either.
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _require(value: str | None, canonical_name: str) -> str:
    if not value:
        raise MissingCredentialError(
            f"'{canonical_name}' is not set. Add it to the repo-root .env file "
            f"(see 14-azure-ai103/README.md for the full variable list)."
        )
    return value


class AzureConfig:
    """Lazily-read accessors for every Azure credential/setting used in this chapter.

    Each property re-reads os.environ on access (cheap) rather than caching at import
    time, so tests/notebooks that monkeypatch or reload .env still see fresh values.
    """

    # ---- Azure AI Foundry project (Agent Service — most sections) ----
    @property
    def project_endpoint(self) -> str:
        return _require(
            _env("AZURE_AI_PROJECT_ENDPOINT", "PROJECT_ENDPOINT"),
            "AZURE_AI_PROJECT_ENDPOINT",
        )

    @property
    def project_resource_id(self) -> str:
        """Full ARM resource ID: /subscriptions/.../projects/<project>."""
        return _require(_env("AZURE_AI_PROJECT_RESOURCE_ID"), "AZURE_AI_PROJECT_RESOURCE_ID")

    @property
    def model_deployment(self) -> str:
        return _env(
            "AZURE_AI_MODEL_DEPLOYMENT", "MODEL_DEPLOYMENT_NAME", "MODEL_DEPLOYMENT",
            default="gpt-5.4",
        )

    def agent_name(self, default: str) -> str:
        """Per-notebook agent names still get their own default (each section deploys
        a differently-named agent), but can be overridden via AZURE_AI_AGENT_NAME
        (or the bare AGENT_NAME some older labs used)."""
        return _env("AZURE_AI_AGENT_NAME", "AGENT_NAME", default=default)

    def model_deployment_or(self, default: str) -> str:
        """Like `model_deployment`, but for the handful of scripts that were built
        against a deliberately different deployment (e.g. a cheaper/faster model)
        than the chapter-wide default."""
        return _env(
            "AZURE_AI_MODEL_DEPLOYMENT", "MODEL_DEPLOYMENT_NAME", "MODEL_DEPLOYMENT",
            default=default,
        )

    @property
    def embedding_deployment(self) -> str:
        """Ch04 LangGraph RAG demo — a text-embedding deployment, distinct from the
        chat deployment above."""
        return _env("AZURE_AI_EMBEDDING_DEPLOYMENT", default="text-embedding-3-small")

    # ---- Direct Azure OpenAI resource (non-Foundry-project: Ch01, Ch04 LangChain) ----
    @property
    def openai_endpoint(self) -> str:
        return _require(_env("AZURE_OPENAI_ENDPOINT"), "AZURE_OPENAI_ENDPOINT")

    @property
    def openai_api_key(self) -> str | None:
        """Optional — prefer Entra ID (DefaultAzureCredential) when this is unset."""
        return _env("AZURE_OPENAI_API_KEY")

    @property
    def openai_deployment(self) -> str:
        return _env("AZURE_OPENAI_DEPLOYMENT", default="gpt-4.1")
    @property
    def openai_deployment2(self) -> str:
        return _env("AZURE_OPENAI_DEPLOYMENT2", default="gpt-5.1")

    # ---- Azure OpenAI image generation deployment (Ch05, Lab 22) ----
    @property
    def image_endpoint(self) -> str:
        return _require(
            _env("AZURE_OPENAI_IMAGE_ENDPOINT", "AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_IMAGE_ENDPOINT",
        )

    @property
    def image_api_key(self) -> str | None:
        return _env("AZURE_OPENAI_IMAGE_API_KEY", "AZURE_OPENAI_API_KEY")

    @property
    def image_deployment(self) -> str:
        return _env(
            "AZURE_OPENAI_IMAGE_DEPLOYMENT", "IMAGE_DEPLOYMENT_NAME",
            default="gpt-image-2",
        )

    # ---- Azure AI Search (Ch02 RAG, Ch08 knowledge base MCP) ----
    @property
    def search_endpoint(self) -> str:
        return _require(
            _env("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_SERVICE_ENDPOINT"),
            "AZURE_SEARCH_ENDPOINT",
        )

    @property
    def search_index_name(self) -> str:
        return _env("AZURE_SEARCH_INDEX_NAME", default="rag-1782198581571")

    @property
    def search_api_key(self) -> str | None:
        """Optional — prefer Entra ID (DefaultAzureCredential) when this is unset."""
        return _env("AZURE_SEARCH_API_KEY")

    @property
    def search_knowledge_base_name(self) -> str:
        return _env("AZURE_SEARCH_KNOWLEDGE_BASE_NAME", default="kb-cloudxeus-course-products")

    def project_mcp_connection_name(self, default: str = "cloudxeus-product-kb-mcp-connection") -> str:
        """A method (not a property) because different scripts register different
        MCP connections in the same Foundry project (Ch08's search-knowledge-base
        connection vs. Ch04's demo Function App connection) — a single shared
        default would make them collide under one connection name."""
        return _env("AZURE_AI_PROJECT_MCP_CONNECTION_NAME", default=default)

    # ---- Azure AI Content Safety (Ch05, Ch08, Lab 23) ----
    @property
    def content_safety_endpoint(self) -> str:
        return _require(
            _env("AZURE_CONTENT_SAFETY_ENDPOINT", "AZURE_CONTENTSAFETY_ENDPOINT"),
            "AZURE_CONTENT_SAFETY_ENDPOINT",
        )

    @property
    def content_safety_key(self) -> str:
        return _require(
            _env("AZURE_CONTENT_SAFETY_KEY", "AZURE_CONTENTSAFETY_KEY"),
            "AZURE_CONTENT_SAFETY_KEY",
        )

    # ---- Azure AI Language (Ch06 NER/PII/sentiment, Lab 12/13) ----
    @property
    def language_endpoint(self) -> str:
        return _require(_env("AZURE_LANGUAGE_ENDPOINT"), "AZURE_LANGUAGE_ENDPOINT")

    @property
    def language_key(self) -> str:
        return _require(_env("AZURE_LANGUAGE_KEY"), "AZURE_LANGUAGE_KEY")

    @property
    def language_resource_name(self) -> str:
        """Bare resource name (no scheme/domain) — needed for MCP URL construction."""
        return _require(_env("AZURE_LANGUAGE_RESOURCE_NAME"), "AZURE_LANGUAGE_RESOURCE_NAME")

    # ---- Azure AI Translator (Ch06, Lab 28 — usually the same multi-service resource as Language) ----
    @property
    def translator_endpoint(self) -> str:
        return _env("AZURE_TRANSLATOR_ENDPOINT", default=self.language_endpoint)

    @property
    def translator_key(self) -> str:
        return _env("AZURE_TRANSLATOR_KEY", default=self.language_key)

    # ---- Azure AI Speech (Ch06, Lab 14/15/16) ----
    @property
    def speech_endpoint(self) -> str:
        return _require(_env("AZURE_SPEECH_ENDPOINT"), "AZURE_SPEECH_ENDPOINT")

    @property
    def speech_key(self) -> str:
        return _require(_env("AZURE_SPEECH_KEY"), "AZURE_SPEECH_KEY")

    @property
    def speech_region(self) -> str:
        return _require(_env("AZURE_SPEECH_REGION"), "AZURE_SPEECH_REGION")

    # ---- Azure AI Content Understanding (Ch07, Ch08, Lab 24) ----
    @property
    def content_understanding_endpoint(self) -> str:
        return _require(
            _env("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "AZURE_CONTENTUNDERSTANDING_ENDPOINT"),
            "AZURE_CONTENT_UNDERSTANDING_ENDPOINT",
        )

    @property
    def content_understanding_key(self) -> str | None:
        """Optional — falls back to DefaultAzureCredential (Entra ID) when unset."""
        return _env("AZURE_CONTENT_UNDERSTANDING_KEY", "AZURE_CONTENTUNDERSTANDING_KEY")

    @property
    def cu_analyzer_id(self) -> str:
        return _env("AZURE_CU_ANALYZER_ID", default="cloudxeusstudentinvoiceanalyzer")

    # ---- Azure AI Document Intelligence (Ch08, Lab 25) ----
    @property
    def document_intelligence_endpoint(self) -> str:
        return _require(
            _env("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "AZURE_DOCUMENTINTELLIGENCE_ENDPOINT"),
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        )

    @property
    def document_intelligence_key(self) -> str:
        return _require(
            _env("AZURE_DOCUMENT_INTELLIGENCE_KEY", "AZURE_DOCUMENTINTELLIGENCE_KEY"),
            "AZURE_DOCUMENT_INTELLIGENCE_KEY",
        )

    # ---- Azure AI Voice Live (Lab 15 real-time voice agent) ----
    @property
    def voicelive_endpoint(self) -> str:
        return _require(_env("AZURE_VOICELIVE_ENDPOINT"), "AZURE_VOICELIVE_ENDPOINT")

    @property
    def voicelive_project_name(self) -> str:
        return _require(_env("AZURE_VOICELIVE_PROJECT_NAME"), "AZURE_VOICELIVE_PROJECT_NAME")

    @property
    def voicelive_agent_id(self) -> str:
        return _require(_env("AZURE_VOICELIVE_AGENT_ID"), "AZURE_VOICELIVE_AGENT_ID")

    # ---- Observability (Ch04 LangChain OpenTelemetry tracing) ----
    @property
    def appinsights_connection_string(self) -> str:
        return _require(
            _env("AZURE_APPINSIGHTS_CONNECTION_STRING"),
            "AZURE_APPINSIGHTS_CONNECTION_STRING",
        )

    # ---- MCP (Ch06/Ch08 MCP tool servers) ----
    @property
    def mcp_system_key(self) -> str | None:
        return _env("MCP_SYSTEM_KEY")

    @property
    def function_app_host(self) -> str:
        """Hostname (no scheme) of a demo Azure Function App used as an MCP tool
        backend (Ch04's demo-functionapp-mcp), e.g. 'my-func.azurewebsites.net'."""
        return _require(_env("AZURE_FUNCTION_APP_HOST"), "AZURE_FUNCTION_APP_HOST")

    # ---- Preview/best-effort labs (Azure OpenAI audio, video, Foundry IQ) ----
    @property
    def audio_deployment(self) -> str:
        """Azure OpenAI audio-capable deployment for TTS/transcription (Lab 14) —
        distinct from the chat deployment above."""
        return _require(
            _env("AZURE_OPENAI_AUDIO_DEPLOYMENT", "MODEL_NAME"),
            "AZURE_OPENAI_AUDIO_DEPLOYMENT",
        )

    @property
    def video_deployment(self) -> str:
        return _env(
            "AZURE_OPENAI_VIDEO_DEPLOYMENT", "VIDEO_DEPLOYMENT_NAME",
            default="sora-2",
        )

    @property
    def knowledge_source_name(self) -> str:
        """Foundry IQ knowledge source name (Lab 19, preview feature)."""
        return _require(
            _env("AZURE_AI_KNOWLEDGE_SOURCE_NAME", "KNOWLEDGE_SOURCE_NAME"),
            "AZURE_AI_KNOWLEDGE_SOURCE_NAME",
        )


config = AzureConfig()
