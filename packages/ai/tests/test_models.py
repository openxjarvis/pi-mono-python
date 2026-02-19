"""Tests for model registry — mirrors packages/ai/test/ model tests."""
import pytest

from pi_ai import get_model, get_models, get_providers, calculate_cost, supports_xhigh, Usage


def test_get_model_anthropic():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    assert model.id == "claude-3-5-sonnet-20241022"
    assert model.provider == "anthropic"
    assert model.api == "anthropic-messages"
    assert model.context_window == 200000


def test_get_model_openai():
    model = get_model("openai", "gpt-4o")
    assert model.id == "gpt-4o"
    assert model.provider == "openai"
    assert model.api == "openai-responses"


def test_get_model_google():
    model = get_model("google", "gemini-2.0-flash")
    assert model.id == "gemini-2.0-flash"
    assert model.provider == "google"
    assert model.api == "google-generative-ai"


def test_get_model_not_found():
    with pytest.raises(KeyError):
        get_model("nonexistent", "fake-model")


def test_get_providers():
    providers = get_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "google" in providers


def test_get_models_all():
    models = get_models()
    assert len(models) >= 700


def test_get_models_by_provider():
    anthropic_models = get_models("anthropic")
    assert all(m.provider == "anthropic" for m in anthropic_models)
    assert len(anthropic_models) > 0


def test_calculate_cost():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    usage = Usage(input=1_000_000, output=500_000, cache_read=0, cache_write=0)
    cost = calculate_cost(model, usage)
    # 1M input * $3/M + 0.5M output * $15/M = $3 + $7.5 = $10.5
    assert abs(cost - 10.5) < 0.01


def test_supports_xhigh_false():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    assert supports_xhigh(model) is False


def test_supports_xhigh_true():
    from pi_ai.types import Model, ModelCost
    model = Model(
        id="gpt-5.2",
        name="GPT-5.2",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32768,
    )
    assert supports_xhigh(model) is True
