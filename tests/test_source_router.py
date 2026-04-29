from rag.config import Config
from rag.pipeline.source_router import plan
from rag.types import FilterSpec, QueryUnderstanding


def test_default_routes_documents_and_kb():
    sp = plan(
        cfg=Config(),
        filters=FilterSpec(),
        understanding=QueryUnderstanding(),
    )
    assert "documents" in sp.routes
    assert "knowledge_base" in sp.routes
    assert sp.use_vector is True


def test_filter_overrides_routes():
    sp = plan(
        cfg=Config(),
        filters=FilterSpec(source_types=["knowledge_base"]),
        understanding=QueryUnderstanding(),
    )
    assert sp.routes == ["knowledge_base"]


def test_unsupported_route_dropped():
    sp = plan(
        cfg=Config(),
        filters=FilterSpec(source_types=["chat"]),  # reserved, not supported
        understanding=QueryUnderstanding(),
    )
    assert "chat" not in sp.routes
    # Falls back to default.
    assert sp.routes == ["documents", "knowledge_base"]


def test_keyword_disabled_when_no_postgres():
    cfg = Config(postgres_url="", enable_hybrid_retrieval=True)
    sp = plan(cfg=cfg, filters=FilterSpec(), understanding=QueryUnderstanding())
    assert sp.use_keyword is False


def test_keyword_forced_for_exact_match():
    cfg = Config(postgres_url="", enable_hybrid_retrieval=False)
    qu = QueryUnderstanding(needs_exact_keyword_match=True)
    sp = plan(cfg=cfg, filters=FilterSpec(), understanding=qu)
    assert sp.use_keyword is True


def test_understanding_source_preference_used_when_filter_empty():
    qu = QueryUnderstanding(source_preference=["knowledge_base"])
    sp = plan(cfg=Config(), filters=FilterSpec(), understanding=qu)
    assert sp.routes == ["knowledge_base"]
