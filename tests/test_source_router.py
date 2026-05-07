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


def test_keyword_enabled_without_postgres_when_hybrid_enabled():
    cfg = Config(postgres_url="", enable_hybrid_retrieval=True)
    sp = plan(cfg=cfg, filters=FilterSpec(), understanding=QueryUnderstanding())
    assert sp.use_keyword is True


def test_keyword_disabled_when_hybrid_disabled():
    cfg = Config(postgres_url="", enable_hybrid_retrieval=False)
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


def test_complex_queries_expand_retrieval_budget():
    cfg = Config(keyword_top_k=10, vector_top_k=10, merged_candidate_limit=20, rerank_top_k=8)
    qu = QueryUnderstanding(query_type="comparison")
    sp = plan(cfg=cfg, filters=FilterSpec(), understanding=qu)
    assert sp.keyword_top_k == 15
    assert sp.vector_top_k == 15
    assert sp.merged_limit == 30
    assert sp.rerank_top_k == 10
