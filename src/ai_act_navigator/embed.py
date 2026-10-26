"""Turn text into embeddings: lists of numbers that capture what the text means.

Texts with similar meaning get similar numbers, even if they use different words.
"Must a chatbot say it's an AI?" lands close to Art. 50(1) ("...informed that they are
interacting with an AI system..."), which is what makes search by meaning possible.

The model runs locally on your machine. The first run downloads it once (about 130 MB)
from Hugging Face into ~/.cache/huggingface. After that, no internet is needed and no
text ever leaves your computer.
"""

from sentence_transformers import SentenceTransformer

# Small, fast, good-quality English model. Swap it here to try another one.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Exact version (git commit) of the model on Hugging Face. Pinning it means we always get
# the same files: results stay reproducible, and a later change to the model repo (even a
# malicious one) can't silently swap what we run. Update it on purpose, never by accident.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Recommended by the model's publisher (BAAI) for short questions that search longer
# passages. The model's own config leaves the query prompt empty, so we add it ourselves.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps the embedding model so the rest of the code doesn't depend on the library.

    Anything with the same two methods (embed_documents, embed_query) can replace it,
    e.g. the FakeEmbedder in the tests, which needs no model download.
    """

    def __init__(self, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION) -> None:
        # e.g. "BAAI/bge-small-en-v1.5@5c38ec7...". The index records it, so search can
        # refuse to compare vectors made by two different models (they'd be meaningless).
        self.model_id = f"{model_name}@{revision}"
        # trust_remote_code=False (the default) means we never run code shipped with the
        # model, we only load its weights. Kept explicit here for security.
        self._model = SentenceTransformer(model_name, revision=revision, trust_remote_code=False)

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
