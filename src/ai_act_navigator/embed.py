"""Turn text into embeddings: lists of numbers that capture what the text means.

Texts with similar meaning get similar numbers, even if they use different words.
"Must a chatbot say it's an AI?" lands close to Art. 50(1) ("...informed that they are
interacting with an AI system..."), which is what makes search by meaning possible.

The model runs locally on your machine, so no text ever leaves your computer.
`ai-act index` downloads it once (about 130 MB) from Hugging Face into
~/.cache/huggingface. `ai-act ask` then loads it from that folder in full offline mode:
no network requests to Hugging Face at all.
"""

import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # only for type hints; the real import happens inside load_model
    from sentence_transformers import CrossEncoder, SentenceTransformer

# Which kind of model to load: an embedding model (embed.py) or a reranker (rerank.py).
ModelKind = Literal["embedding", "reranker"]

# Small, fast, good-quality English model. Swap it here to try another one.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Exact version (git commit) of the model on Hugging Face. Pinning it means we always get
# the same files: results stay reproducible, and a later change to the model repo (even a
# malicious one) can't silently swap what we run. Update it on purpose, never by accident.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Recommended by the model's publisher (BAAI) for short questions that search longer
# passages. The model's own config leaves the query prompt empty, so we add it ourselves.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class ModelNotAvailableError(Exception):
    """A local model (embedding or reranker) is not on this computer and couldn't be downloaded."""


def load_model(
    model_name: str, revision: str, allow_download: bool, kind: ModelKind = "embedding"
) -> "SentenceTransformer | CrossEncoder":
    """Load the model from the local cache; download it only if allowed and not there yet.

    Used for both local models, so the offline rules below live in one place: the embedding
    model (a SentenceTransformer) and the reranker (a CrossEncoder, see rerank.py).

    allow_download=False is full offline mode: no network requests to Hugging Face at all.
    local_files_only alone isn't enough, because the library still makes small requests of
    its own (e.g. about once a day it fetches a list of known AI coding tools from Hugging
    Face). HF_HUB_OFFLINE=1 switches those off too, but the library reads it only once, when
    it is first imported. That's why sentence_transformers is imported here, after setting
    it, and not at the top of the file. (In a program that already imported it, only
    local_files_only applies. `ai-act ask` imports it here first; checked by hand.)

    allow_download=True first tries the cache (0.4 s instead of over a minute without
    internet), then downloads. The pinned revision never changes, so a cached copy is
    always the right one.
    """
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from sentence_transformers import CrossEncoder, SentenceTransformer

    model_class = SentenceTransformer if kind == "embedding" else CrossEncoder
    # trust_remote_code=False (the default) means we never run code shipped with the
    # model, we only load its weights. Kept explicit here for security.
    try:
        return model_class(
            model_name, revision=revision, trust_remote_code=False, local_files_only=True
        )
    except OSError as error:  # not in the cache yet
        if not allow_download:
            raise ModelNotAvailableError(
                f"The {kind} model isn't downloaded yet. Run: uv run ai-act index"
            ) from error
    try:
        return model_class(model_name, revision=revision, trust_remote_code=False)
    except OSError as error:  # e.g. no internet
        raise ModelNotAvailableError(
            f"Could not download the {kind} model {model_name} from Hugging Face. "
            "Check your internet connection."
        ) from error


class Embedder:
    """Wraps the embedding model so the rest of the code doesn't depend on the library.

    Anything with the same two methods (embed_documents, embed_query) can replace it,
    e.g. the FakeEmbedder in the tests, which needs no model download.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        allow_download: bool = True,
    ) -> None:
        # e.g. "BAAI/bge-small-en-v1.5@5c38ec7...". The index records it, so search can
        # refuse to compare vectors made by two different models (they'd be meaningless).
        self.model_id = f"{model_name}@{revision}"
        self._model = load_model(model_name, revision, allow_download)

    @property
    def max_tokens(self) -> int:
        """Longest input the model reads. Anything longer is cut off (512 for this model)."""
        return self._model.max_seq_length

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeddings for the texts we store and search through (the chunks)."""
        vectors = self._model.encode_document(
            texts,
            batch_size=32,  # how many texts the model processes at once
            normalize_embeddings=True,  # length 1, so cosine similarity = a simple dot product
            show_progress_bar=True,
        )
        return vectors.tolist()  # numpy array -> plain Python lists

    def embed_query(self, text: str) -> list[float]:
        """Embedding for a user's question.

        Separate from embed_documents because questions get QUERY_INSTRUCTION in front:
        a short question and a long legal paragraph look different, and the instruction
        tells the model to match them by meaning anyway.
        """
        return self._model.encode(QUERY_INSTRUCTION + text, normalize_embeddings=True).tolist()

    def count_truncated(self, texts: list[str]) -> int:
        """How many texts are longer than max_tokens, so their end is ignored by the model.

        "Tokens" are the word pieces the model reads: roughly 4 characters of English each.
        """
        tokenizer = self._model.tokenizer
        return sum(len(tokenizer.encode(text, verbose=False)) > self.max_tokens for text in texts)
