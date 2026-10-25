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


class Embedder:
    """Wraps the embedding model so the rest of the code doesn't depend on the library.

    Anything with the same two methods (embed_documents, embed_query) can replace it,
    e.g. the FakeEmbedder in the tests, which needs no model download.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # trust_remote_code=False (the default) means we never run code shipped with the
        # model, we only load its weights. Kept explicit here for security.
        self._model = SentenceTransformer(model_name, trust_remote_code=False)

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

        Separate from embed_documents because some models (like this one) add a short
        instruction to questions so they match the right passages better.
        """
        return self._model.encode_query(text, normalize_embeddings=True).tolist()

    def count_truncated(self, texts: list[str]) -> int:
        """How many texts are longer than max_tokens, so their end is ignored by the model.

        "Tokens" are the word pieces the model reads: roughly 4 characters of English each.
        """
        tokenizer = self._model.tokenizer
        return sum(len(tokenizer.encode(text, verbose=False)) > self.max_tokens for text in texts)
