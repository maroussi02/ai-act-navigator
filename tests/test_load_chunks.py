"""UNIT TEST for saving and loading chunks (the data/chunks.jsonl format)."""

from ai_act_navigator.chunk import load_chunks, save_chunks
from ai_act_navigator.models import Chunk


def test_saved_chunks_load_back_unchanged(tmp_path):
    chunks = [
        Chunk("art-50-1", "article", "Art. 50(1)", "Transparency", "‘AI system’ means…", "u"),
        Chunk("rct-1", "recital", "Recital 1", "", "(1) The purpose…", "u"),
    ]
    path = tmp_path / "chunks.jsonl"

    save_chunks(chunks, path)

    assert load_chunks(path) == chunks
