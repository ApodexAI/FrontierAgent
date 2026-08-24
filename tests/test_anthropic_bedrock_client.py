from __future__ import annotations

import httpx2
import pytest

from frontier_agent.infra.anthropic_client import _build_bedrock_client


@pytest.mark.asyncio
async def test_bedrock_client_injects_bearer_auth_with_locked_transport() -> None:
    client = _build_bedrock_client(
        "bedrock-test-token",
        "https://bedrock-runtime.us-east-1.amazonaws.com",
        5.0,
    )
    request = httpx2.Request(
        "POST",
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
    )

    try:
        await client._prepare_request(request)
    finally:
        await client.close()

    assert request.headers["Authorization"] == "Bearer bedrock-test-token"
