"""Anthropic Claude API wrapper used by run_experiments.py."""

import os
import time
from typing import Optional

CLAUDE_MODELS = {
    "claude-sonnet-4-6": {
        "api_id":     "claude-sonnet-4-6",
        "short_name": "Claude Sonnet 4.6",
        "display":    "Claude Sonnet 4.6",
        "max_tokens": 512,
        "type":       "claude",
    },
}

CLAUDE_MODEL_KEYS = list(CLAUDE_MODELS.keys())


def is_claude_model(model_key: str) -> bool:
    return model_key in CLAUDE_MODELS


def load_claude_client(api_key: str = None):
    """Return an anthropic.Anthropic client. Reads ANTHROPIC_API_KEY env var by default."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "Anthropic API key not found. Pass --claude-api-key or "
            "set ANTHROPIC_API_KEY environment variable."
        )
    return anthropic.Anthropic(api_key=key)


def _split_messages(messages):
    """Split chat-template messages into (system_text, user_assistant_list)."""
    sys_parts = []
    rest = []
    for m in messages:
        role = m["role"]
        if role == "system":
            sys_parts.append(m["content"])
        elif role == "model":
            rest.append({"role": "assistant", "content": m["content"]})
        else:
            rest.append({"role": role, "content": m["content"]})
    return "\n\n".join(sys_parts), rest


def run_claude_inference(
    client,
    messages: list[dict],
    model_key: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[str, float]:
    """Single-turn Claude call. Returns (raw_text, latency)."""
    cfg = CLAUDE_MODELS[model_key]
    system_text, chat_msgs = _split_messages(messages)

    system_blocks = [{
        "type": "text",
        "text": system_text or "You are a helpful assistant.",
        "cache_control": {"type": "ephemeral"},
    }]

    t0 = time.time()
    last_error = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=cfg["api_id"],
                max_tokens=cfg["max_tokens"],
                system=system_blocks,
                messages=chat_msgs,
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return text.strip(), time.time() - t0
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                time.sleep(retry_delay * (2 ** attempt))
            elif "500" in err_str or "503" in err_str or "overloaded" in err_str or "529" in err_str:
                time.sleep(retry_delay)
            elif "401" in err_str or "400" in err_str:
                raise
            else:
                time.sleep(retry_delay)
    raise RuntimeError(f"Claude inference failed after {retries} attempts: {last_error}")


def run_claude_inference_cot(
    client,
    messages: list[dict],
    model_key: str,
    cot_extract_instruction: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[str, str, float]:
    """Two-turn CoT inference. Turn 1 produces reasoning; turn 2 extracts JSON."""
    cfg = CLAUDE_MODELS[model_key]
    system_text, chat_msgs = _split_messages(messages)
    system_blocks = [{
        "type": "text",
        "text": system_text or "You are a helpful assistant.",
        "cache_control": {"type": "ephemeral"},
    }]

    t0 = time.time()

    last_error = None
    for attempt in range(retries):
        try:
            r1 = client.messages.create(
                model=cfg["api_id"],
                max_tokens=cfg["max_tokens"],
                system=system_blocks,
                messages=chat_msgs,
            )
            cot_text = next((b.text for b in r1.content if b.type == "text"), "").strip()
            break
        except Exception as e:
            last_error = e
            time.sleep(retry_delay * (2 ** attempt) if "429" in str(e) else retry_delay)
    else:
        raise RuntimeError(f"Claude CoT turn 1 failed: {last_error}")

    chat_msgs_t2 = chat_msgs + [
        {"role": "assistant", "content": cot_text},
        {"role": "user", "content": cot_extract_instruction},
    ]
    last_error = None
    for attempt in range(retries):
        try:
            r2 = client.messages.create(
                model=cfg["api_id"],
                max_tokens=cfg["max_tokens"],
                system=system_blocks,
                messages=chat_msgs_t2,
            )
            json_text = next((b.text for b in r2.content if b.type == "text"), "").strip()
            return cot_text, json_text, time.time() - t0
        except Exception as e:
            last_error = e
            time.sleep(retry_delay * (2 ** attempt) if "429" in str(e) else retry_delay)
    raise RuntimeError(f"Claude CoT turn 2 failed: {last_error}")


def claude_cost_estimate(n_scenarios: int, model_key: str,
                          n_strategies: int = 3, cot: bool = False) -> str:
    """Rough cost estimate for planning."""
    pricing = {
        "claude-sonnet-4-6": (0.003, 0.015),  # $/1K tokens (in, out)
    }
    p_in, p_out = pricing.get(model_key, (0.003, 0.015))
    total_calls = n_scenarios * n_strategies
    if cot:
        in_tokens = total_calls * (800 + 1200)
        out_tokens = total_calls * (400 + 150)
    else:
        in_tokens = total_calls * 800
        out_tokens = total_calls * 150
    cost = (in_tokens * p_in + out_tokens * p_out) / 1000
    label = " (CoT)" if cot else ""
    return (f"~{total_calls * (2 if cot else 1)} API calls{label} | "
            f"~{in_tokens // 1000}K in + {out_tokens // 1000}K out tokens | "
            f"est. ${cost:.2f} (excl. cache savings)")
