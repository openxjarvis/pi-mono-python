"""
Tests for the OAuth subpackage.

Tests PKCE generation, provider registry, and token refresh (mocked HTTP).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------

class TestPKCE:
    def test_generate_pkce_returns_two_strings(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, challenge = generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_pkce_verifier_length(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, _ = generate_pkce()
        # 32 bytes base64url encoded = 43 chars (without padding)
        assert len(verifier) >= 40

    def test_pkce_challenge_length(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        _, challenge = generate_pkce()
        assert len(challenge) >= 40

    def test_pkce_is_different_each_call(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        v1, c1 = generate_pkce()
        v2, c2 = generate_pkce()
        assert v1 != v2
        assert c1 != c2

    def test_pkce_base64url_no_padding(self):
        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, challenge = generate_pkce()
        assert "=" not in verifier
        assert "=" not in challenge

    def test_pkce_challenge_is_sha256_of_verifier(self):
        """Verify challenge is correct SHA-256 base64url of verifier."""
        import base64
        import hashlib

        from pi_ai.utils.oauth.pkce import generate_pkce
        verifier, challenge = generate_pkce()
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        assert challenge == expected


# ---------------------------------------------------------------------------
# OAuth Types
# ---------------------------------------------------------------------------

class TestOAuthCredentials:
    def test_to_dict_round_trip(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        creds = OAuthCredentials(refresh="r", access="a", expires=12345)
        d = creds.to_dict()
        assert d["refresh"] == "r"
        assert d["access"] == "a"
        assert d["expires"] == 12345

    def test_from_dict(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        d = {"refresh": "r", "access": "a", "expires": 99999}
        creds = OAuthCredentials.from_dict(d)
        assert creds.refresh == "r"
        assert creds.access == "a"
        assert creds.expires == 99999

    def test_from_dict_with_extra_fields(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        d = {"refresh": "r", "access": "a", "expires": 0, "account_id": "u123"}
        creds = OAuthCredentials.from_dict(d)
        assert creds.extra.get("account_id") == "u123"


# ---------------------------------------------------------------------------
# OAuth Registry
# ---------------------------------------------------------------------------

class TestOAuthRegistry:
    def test_get_oauth_provider_anthropic(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("anthropic")
        assert p is not None
        assert p.id == "anthropic"
        assert p.name

    def test_get_oauth_provider_github_copilot(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("github-copilot")
        assert p is not None

    def test_get_oauth_provider_gemini_cli(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("google-gemini-cli")
        assert p is not None

    def test_get_oauth_provider_antigravity(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("google-antigravity")
        assert p is not None

    def test_get_oauth_provider_openai_codex(self):
        from pi_ai.utils.oauth import get_oauth_provider
        p = get_oauth_provider("openai-codex")
        assert p is not None

    def test_get_unknown_provider_returns_none(self):
        from pi_ai.utils.oauth import get_oauth_provider
        assert get_oauth_provider("nonexistent-provider") is None

    def test_get_oauth_providers_returns_all(self):
        from pi_ai.utils.oauth import get_oauth_providers
        providers = get_oauth_providers()
        assert len(providers) >= 5
        ids = [p.id for p in providers]
        assert "anthropic" in ids
        assert "github-copilot" in ids

    def test_register_custom_provider(self):
        from pi_ai.utils.oauth import get_oauth_provider, register_oauth_provider

        class _Custom:
            id = "custom-test-provider-xyz"
            name = "Custom Test"
            uses_callback_server = False

            async def login(self, callbacks): ...
            async def refresh_token(self, credentials): return credentials
            def get_api_key(self, credentials): return credentials.access
            def modify_models(self, models, credentials): return models

        register_oauth_provider(_Custom())
        p = get_oauth_provider("custom-test-provider-xyz")
        assert p is not None
        assert p.name == "Custom Test"


# ---------------------------------------------------------------------------
# get_oauth_api_key
# ---------------------------------------------------------------------------

class TestGetOAuthApiKey:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_credentials(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        result = await get_oauth_api_key("anthropic", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_api_key_when_credentials_valid(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        future_ts = int(time.time() * 1000) + 3600 * 1000  # 1 hour from now
        creds_dict = {"refresh": "r", "access": "valid_token", "expires": future_ts}
        result = await get_oauth_api_key("anthropic", {"anthropic": creds_dict})
        assert result is not None
        assert result["api_key"] == "valid_token"

    @pytest.mark.asyncio
    async def test_refreshes_expired_token(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        from pi_ai.utils.oauth.types import OAuthCredentials

        # Expired credentials
        expired_creds = {"refresh": "r", "access": "old_token", "expires": 0}
        new_creds = OAuthCredentials(refresh="new_r", access="new_token", expires=int(time.time() * 1000) + 3600000)

        with patch("pi_ai.utils.oauth.get_oauth_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.refresh_token = AsyncMock(return_value=new_creds)
            mock_provider.get_api_key = lambda c: c.access
            mock_get.return_value = mock_provider

            result = await get_oauth_api_key("anthropic", {"anthropic": expired_creds})
            assert result is not None
            assert result["api_key"] == "new_token"

    @pytest.mark.asyncio
    async def test_raises_for_unknown_provider(self):
        from pi_ai.utils.oauth import get_oauth_api_key
        with pytest.raises(ValueError, match="Unknown OAuth provider"):
            await get_oauth_api_key("fake-provider", {})


# ---------------------------------------------------------------------------
# Anthropic OAuth Provider (mocked HTTP)
# ---------------------------------------------------------------------------

class TestAnthropicOAuthProvider:
    def test_provider_id(self):
        from pi_ai.utils.oauth.anthropic import anthropic_oauth_provider
        assert anthropic_oauth_provider.id == "anthropic"

    def test_get_api_key(self):
        from pi_ai.utils.oauth.types import OAuthCredentials
        from pi_ai.utils.oauth.anthropic import anthropic_oauth_provider
        creds = OAuthCredentials(refresh="r", access="tok_123", expires=99999)
        assert anthropic_oauth_provider.get_api_key(creds) == "tok_123"

    @pytest.mark.asyncio
    async def test_refresh_token_mocked(self):
        from pi_ai.utils.oauth.anthropic import refresh_anthropic_token

        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        }

        with patch("pi_ai.utils.oauth.anthropic.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_ctx

            creds = await refresh_anthropic_token("old_refresh")
            assert creds.access == "new_access"
            assert creds.refresh == "new_refresh"


# ---------------------------------------------------------------------------
# GitHub Copilot OAuth (unit tests)
# ---------------------------------------------------------------------------

class TestGitHubCopilotOAuth:
    def test_normalize_domain(self):
        from pi_ai.utils.oauth.github_copilot import normalize_domain
        assert normalize_domain("github.example.com") == "github.example.com"
        assert normalize_domain("https://github.example.com") == "github.example.com"
        assert normalize_domain("  github.com  ") == "github.com"

    def test_get_base_url_default(self):
        from pi_ai.utils.oauth.github_copilot import get_github_copilot_base_url
        assert get_github_copilot_base_url() == "https://api.github.com"

    def test_get_base_url_custom_domain(self):
        from pi_ai.utils.oauth.github_copilot import get_github_copilot_base_url
        url = get_github_copilot_base_url("my-corp.github.com")
        assert "my-corp.github.com" in url
