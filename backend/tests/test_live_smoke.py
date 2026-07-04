"""One-off diagnostic against the real OpenAI API and the real ingested
chroma_db, to distinguish a code bug from an environment/API-key issue.

Never runs by default (see pyproject.toml's `addopts = "-m 'not live'"`).
Run manually with: uv run pytest backend/tests -m live -s
"""
import sys
import traceback
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import Config
from rag_system import RAGSystem

pytestmark = pytest.mark.live


def test_live_content_question_does_not_crash():
    base_config = Config()
    if not base_config.OPENAI_API_KEY or len(base_config.OPENAI_API_KEY) < 20:
        pytest.skip("No valid OPENAI_API_KEY configured for live smoke test")

    config = Config(
        OPENAI_API_KEY=base_config.OPENAI_API_KEY,
        CHROMA_PATH=str(BACKEND_DIR / "chroma_db"),
    )
    rag_system = RAGSystem(config)

    if rag_system.vector_store.get_course_count() == 0:
        pytest.skip("No courses ingested in backend/chroma_db - run the app once to ingest ../docs first")

    try:
        answer, sources = rag_system.query("What is covered in lesson 1?")
    except Exception as e:
        pytest.fail(f"rag_system.query() raised {type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    assert isinstance(answer, str) and answer.strip()
    print(f"\nLive answer: {answer}\nSources: {sources}")
