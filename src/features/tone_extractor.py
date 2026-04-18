"""
Phase 2C: Claude Tone Feature Extractor
Calls Claude API to extract distress tone features from earnings call transcripts.
Caches all results to avoid redundant API calls.
"""

import os
import sys
import json
import time
import logging
import anthropic
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import TRANSCRIPTS_DIR, TONE_CACHE_DIR, PROCESSED_DIR, CLAUDE_MODEL, MAX_TRANSCRIPT_WORDS
from src.utils.helpers import logger, update_progress

TONE_FIELDS = [
    "evasiveness_score", "confidence_score", "debt_mention_count",
    "reassurance_score", "analyst_tension_score", "guidance_vagueness_score",
    "liquidity_stress_mentions", "tone_shift_flag", "overall_distress_score",
    "key_phrases",
]

SYSTEM_PROMPT = (
    "You are a financial analyst specializing in corporate distress detection. "
    "Analyze the CEO/MD speech from this Indian company earnings call. "
    "Return ONLY a valid JSON object with no other text, no markdown, no explanation. "
    "Return exactly these fields: "
    "evasiveness_score (0-10, how often CEO deflects or avoids direct answers), "
    "confidence_score (0-10, forward-looking positive language), "
    "debt_mention_count (integer, how many times debt loans pledging borrowing is mentioned), "
    "reassurance_score (0-10, over-reassurance language like everything is fine or no issues), "
    "analyst_tension_score (0-10, tension or hostility in Q&A responses), "
    "guidance_vagueness_score (0-10, vagueness about future plans or guidance), "
    "liquidity_stress_mentions (integer, mentions of cash flow liquidity working capital), "
    "tone_shift_flag (0 or 1, whether tone in Q&A section differs noticeably from prepared statement), "
    "overall_distress_score (0-10, holistic financial distress signal), "
    "key_phrases (list of up to 5 phrases that most drove your assessment)."
)


def truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def call_claude_tone(client: anthropic.Anthropic, text: str) -> dict:
    """Call Claude API and parse JSON response. Retry once on parse failure."""
    truncated = truncate_to_words(text, MAX_TRANSCRIPT_WORDS)

    messages = [{"role": "user", "content": truncated}]

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            return parsed

        except json.JSONDecodeError:
            if attempt == 0:
                logger.warning("JSON parse failed, retrying with explicit instruction...")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Your response was not valid JSON. Return ONLY the JSON object, no markdown, no explanation."})
            else:
                logger.error(f"Claude JSON parse failed twice: {raw[:200]}")
                return {}
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            if "rate_limit" in str(e).lower():
                time.sleep(60)
            return {}

    return {}


def load_transcript(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def run_phase_2c():
    logger.info("=== PHASE 2C: Claude Tone Features ===")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set in environment. Cannot run tone extraction.")
        update_progress("FAILED", "Phase 2C: Claude Tone Features", "ANTHROPIC_API_KEY missing")
        return pd.DataFrame()

    client = anthropic.Anthropic(api_key=api_key)

    transcript_files = list(TRANSCRIPTS_DIR.glob("*.json"))
    logger.info(f"Found {len(transcript_files)} transcript files")

    rows = []
    api_calls = 0
    cache_hits = 0

    for tf in transcript_files:
        stem = tf.stem  # e.g. "DHFL_2019Q3"
        cache_path = TONE_CACHE_DIR / f"{stem}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                tone_data = json.load(f)
            cache_hits += 1
        else:
            transcript = load_transcript(tf)
            if not transcript:
                continue

            # Prefer qa_ceo_text, fall back to prepared_text, then full_text
            text = (transcript.get("qa_ceo_text") or
                    transcript.get("prepared_text") or
                    transcript.get("full_text") or "")
            if len(text.split()) < 100:
                text = transcript.get("full_text", "")

            if len(text.strip()) < 200:
                continue

            tone_data = call_claude_tone(client, text)
            api_calls += 1

            if tone_data:
                tone_data["symbol"] = transcript.get("symbol", "")
                tone_data["quarter"] = transcript.get("quarter", "")
                with open(cache_path, "w") as f:
                    json.dump(tone_data, f, indent=2)
                time.sleep(0.5)

        if tone_data:
            # Parse symbol/quarter from filename
            parts = stem.rsplit("_", 1)
            symbol = parts[0] if len(parts) == 2 else stem
            quarter = parts[1] if len(parts) == 2 else "UNKNOWN"

            row = {"nse_symbol": symbol, "quarter": quarter, "tone_available": 1}
            for field in TONE_FIELDS:
                row[field] = tone_data.get(field, -1 if "score" in field or "count" in field or "mentions" in field or "flag" in field else [])
            # key_phrases as JSON string
            if isinstance(row.get("key_phrases"), list):
                row["key_phrases"] = json.dumps(row["key_phrases"])
            rows.append(row)

    df_tone = pd.DataFrame(rows)

    # Ensure all numeric tone fields are numeric
    numeric_tone_cols = [f for f in TONE_FIELDS if f != "key_phrases"]
    for col in numeric_tone_cols:
        if col in df_tone.columns:
            df_tone[col] = pd.to_numeric(df_tone[col], errors="coerce").fillna(-1)

    out_path = PROCESSED_DIR / "tone_features.parquet"
    df_tone.to_parquet(out_path, index=False)

    note = (f"Tone features for {len(df_tone)} transcripts; "
            f"{api_calls} API calls, {cache_hits} cache hits")
    update_progress("DONE", "Phase 2C: Claude Tone Features", note)
    logger.info(f"Phase 2C complete. {note}")
    return df_tone


if __name__ == "__main__":
    run_phase_2c()
