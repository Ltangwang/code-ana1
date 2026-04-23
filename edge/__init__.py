"""Edge module: local Ollama inference and optional Java DFG helpers."""

from .java_dfg_skeleton import extract_java_dfg_skeleton
from .local_inference import OllamaInference

__all__ = ["OllamaInference", "extract_java_dfg_skeleton"]
