"""openai_inference.py"""

import os
import re
import time
import json


OPENAI_MODELS = {
    "gpt-5.4": {
        "api_id":      "gpt-5.4",
        "short_name":  "GPT-5.4",
        "display":     "GPT-5.4",
        "max_tokens":  512,
        "temperature": 0.0,
        "type":        "closed",
    },
    "gpt-4o": {
        "api_id":      "gpt-4o",
        "short_name":  "GPT-4o",
        "display":     "GPT-4o",
        "max_tokens":  512,
        "temperature": 0.0,
        "type":        "closed",
    },
    "gpt-4o-mini": {
        "api_id":      "gpt-4o-mini",
        "short_name":  "GPT-4o-mini",
        "display":     "GPT-4o-mini",
        "max_tokens":  512,
        "temperature": 0.0,
        "type":        "closed",
    },
}

OPENAI_MODEL_KEYS = list(OPENAI_MODELS.keys())


def is_openai_model(model_key: str) -> bool:
    return model_key in OPENAI_MODELS


def load_openai_client(api_key: str = None):
    """Return an openai.OpenAI client."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found. Pass --openai-api-key or "
            "set OPENAI_API_KEY environment variable."
        )
    return openai.OpenAI(api_key=key)


def run_openai_inference(
    client,
    messages: list[dict],
    model_key: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[str, float]:
    """Call the OpenAI Chat Completions API."""
    cfg = OPENAI_MODELS[model_key]

    # OpenAI only accepts 'system', 'user', 'assistant' roles
    # Convert any 'model' role (Gemma convention) → 'assistant'
    clean_messages = []
    for m in messages:
        role = m["role"]
        if role == "model":
            role = "assistant"
        clean_messages.append({"role": role, "content": m["content"]})

    t0 = time.time()
    last_error = None

    # GPT-5.4 and newer OpenAI models use max_completion_tokens, not max_tokens
    token_param = ("max_completion_tokens"
                   if cfg["api_id"].startswith("gpt-5")
                   else "max_tokens")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model    = cfg["api_id"],
                messages = clean_messages,
                **{token_param: cfg["max_tokens"]},
                temperature = cfg["temperature"],
            )
            raw = response.choices[0].message.content.strip()
            return raw, time.time() - t0

        except Exception as e:
            last_error = e
            err_str    = str(e).lower()

            # Rate limit — wait longer
            if "rate_limit" in err_str or "429" in err_str:
                wait = retry_delay * (2 ** attempt)
                time.sleep(wait)
            # Server error — short wait
            elif "500" in err_str or "503" in err_str or "timeout" in err_str:
                time.sleep(retry_delay)
            # Auth / bad request — no point retrying
            elif "401" in err_str or "400" in err_str:
                raise
            else:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"OpenAI inference failed after {retries} attempts: {last_error}"
    )


def openai_cost_estimate(n_scenarios: int, model_key: str,
                          n_strategies: int = 3, cot: bool = False) -> str:
    """Rough token-cost estimate for planning purposes."""
    pricing = {
        "gpt-5.4":     (0.010,  0.030),
        "gpt-4o":      (0.0025, 0.010),
        "gpt-4o-mini": (0.00015,0.0006),
    }
    p_in, p_out = pricing.get(model_key, (0.002, 0.008))
    total_calls  = n_scenarios * n_strategies
    if cot:
        # Two API calls per scenario: reasoning + extraction
        in_tokens  = total_calls * (800 + 1200)   # t1 input + t2 input (longer)
        out_tokens = total_calls * (400 + 150)     # reasoning + JSON
    else:
        in_tokens  = total_calls * 800
        out_tokens = total_calls * 150
    cost = (in_tokens * p_in + out_tokens * p_out) / 1000
    label = " (CoT)" if cot else ""
    return (f"~{total_calls * (2 if cot else 1)} API calls{label} | "
            f"~{in_tokens//1000}K in + {out_tokens//1000}K out tokens | "
            f"est. ${cost:.2f}")


def run_openai_inference_cot(
    client,
    messages: list[dict],   # first-turn messages (includes COT_THINK_INSTRUCTION)
    model_key: str,
    cot_extract_instruction: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[str, str, float]:
    """Two-turn CoT inference for OpenAI models."""
    cfg = OPENAI_MODELS[model_key]
    token_param = ("max_completion_tokens"
                   if cfg["api_id"].startswith("gpt-5")
                   else "max_tokens")

    def _clean(msgs):
        return [{"role": ("assistant" if m["role"] == "model" else m["role"]),
                 "content": m["content"]} for m in msgs]

    t0 = time.time()

    cot_text = ""
    last_error = None
    for attempt in range(retries):
        try:
            r1 = client.chat.completions.create(
                model=cfg["api_id"],
                messages=_clean(messages),
                **{token_param: 512},
                temperature=cfg["temperature"],
            )
            cot_text = r1.choices[0].message.content.strip()
            break
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                time.sleep(retry_delay * (2 ** attempt))
            elif "401" in err_str or "400" in err_str:
                raise
            else:
                time.sleep(retry_delay)
    else:
        raise RuntimeError(f"CoT turn-1 failed after {retries} attempts: {last_error}")

    messages_t2 = _clean(messages) + [
        {"role": "assistant", "content": cot_text},
        {"role": "user",      "content": cot_extract_instruction},
    ]
    json_text = ""
    for attempt in range(retries):
        try:
            r2 = client.chat.completions.create(
                model=cfg["api_id"],
                messages=messages_t2,
                **{token_param: 150},
                temperature=cfg["temperature"],
            )
            json_text = r2.choices[0].message.content.strip()
            break
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                time.sleep(retry_delay * (2 ** attempt))
            elif "401" in err_str or "400" in err_str:
                raise
            else:
                time.sleep(retry_delay)
    else:
        raise RuntimeError(f"CoT turn-2 failed after {retries} attempts: {last_error}")

    return cot_text, json_text, time.time() - t0


def score_classes_openai(
    client,
    messages: list[dict],
    model_key: str,
    classes: list[str],
    retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Score each class by running a short constrained completion with logprobs."""
    cfg = OPENAI_MODELS[model_key]

    # GPT-5.x: max_completion_tokens; older: max_tokens
    token_param = ("max_completion_tokens"
                   if cfg["api_id"].startswith("gpt-5")
                   else "max_tokens")

    # Build the constrained-final-answer messages
    clean = []
    for m in messages:
        role = m["role"]
        if role == "model":
            role = "assistant"
        clean.append({"role": role, "content": m["content"]})
    instruction = (
        "Output exactly one word and nothing else, chosen from this list: "
        + ", ".join(classes) + "."
    )
    clean = clean + [{"role": "user", "content": instruction}]

    last_error = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=cfg["api_id"],
                messages=clean,
                **{token_param: 4},
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )
            content = resp.choices[0].logprobs.content
            if not content:
                # Fallback: equal probability
                return {c: 1.0 / len(classes) for c in classes}

            scores = None
            classes_lc = [c.lower() for c in classes]
            for tok in content[:3]:
                tok_str = tok.token.strip().strip('"').lower()
                if any(tok_str.startswith(c[:3]) for c in classes_lc):
                    cand = {}
                    for alt in tok.top_logprobs:
                        a = alt.token.strip().strip('"').lower()
                        for c, c_lc in zip(classes, classes_lc):
                            if a.startswith(c_lc[:3]) and c not in cand:
                                cand[c] = alt.logprob
                    if cand:
                        scores = cand
                        break

            if scores is None:
                return {}

            # Softmax-normalize over the classes we found logprobs for
            import math
            for c in classes:
                if c not in scores:
                    scores[c] = -20.0  # very low prob for missing classes
            mx = max(scores.values())
            exps = {c: math.exp(scores[c] - mx) for c in classes}
            total = sum(exps.values())
            return {c: exps[c] / total for c in classes}

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                time.sleep(retry_delay * (2 ** attempt))
            elif "401" in err_str or "400" in err_str:
                # Some OpenAI models reject logprobs entirely — return
                # empty so caller falls back to confidence-based scoring.
                if "logprobs" in err_str or "unsupported" in err_str:
                    return {}
                raise
            else:
                time.sleep(retry_delay)
    # All retries failed — return empty (caller falls back).
    return {}
