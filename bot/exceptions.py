"""Custom exception types for bot runtime failures."""


class BotBaseError(Exception):
    """Base class for bot-specific exceptions."""


class RAGPipelineError(BotBaseError):
    """Raised for RAG retrieval or generation pipeline failures."""


class DocumentProcessingError(BotBaseError):
    """Raised when uploaded document processing fails."""


class RateLimitExceededError(BotBaseError):
    """Raised when user exceeds configured rate limits."""


class InsufficientContextError(BotBaseError):
    """Raised when retrieval context is insufficient for a grounded answer."""


class AuthorizationError(BotBaseError):
    """Raised when an unauthorized user attempts restricted actions."""
