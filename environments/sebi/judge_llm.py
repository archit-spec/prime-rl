import hashlib
import json
import logging
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional

import openai

logger = logging.getLogger(__name__)


class JudgeLLM:

    DEFAULT_MODEL = "private-large"
    DEFAULT_BASE_URL = "https://grid.ai.juspay.net/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "private-large",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache_path: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        reward_mode: str = "zero_to_one",
    ):
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "")
        self.base_url = base_url or os.environ.get("JUDGE_BASE_URL", self.DEFAULT_BASE_URL)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        rank = int(os.environ.get("RANK", "0") or 0)
        if self.cache_path and world_size > 1:
            root, ext = os.path.splitext(self.cache_path)
            self.cache_path = f"{root}.rank{rank}{ext or '.pkl'}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.reward_mode = reward_mode
        self.cache: dict = {}

        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.cache = pickle.load(f)

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            # We do explicit retry handling below. Disable SDK-level retries
            # to avoid compounded backoff + long tail latency.
            max_retries=0,
        )

    def _cache_key(self, question: str, gold: str, generated: str) -> str:
        text = f"{question}||{gold}||{generated}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _precheck_penalty(self, generated_answer: str) -> Optional[dict]:
        text = (generated_answer or "").strip()

        if not text:
            return {
                "correctness": 0.0,
                "completeness": 0.0,
                "citation_quality": 0.0,
                "hallucination_penalty": 0.0,
                "format_penalty": 1.0,
                "clarity": 0.0,
                "final_score": 0.05,
                "reason": "empty generated answer",
            }

        if len(text.split()) < 3:
            return {
                "correctness": 0.0,
                "completeness": 0.0,
                "citation_quality": 0.0,
                "hallucination_penalty": 0.0,
                "format_penalty": 0.8,
                "clarity": 0.2,
                "final_score": 0.05,
                "reason": "generated answer is too short to be useful",
            }

        return None

    def _build_prompt(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> str:
        return f"""You are a strict scoring engine for RAG evaluation.

Evaluate Agentic_answer against Answer for the same Question.

Hard rules:
1. Return ONLY valid JSON.
2. Return exactly ONE object (no array) with this exact shape:
{{
  "score": {{
    "Factuality": <integer 1-10>,
    "Completeness": <integer 1-10>,
    "Overall_Score": <integer 1-10>,
    "Reason": "<single concise sentence in format: Factuality=X because ... Completeness=Y because ...>",
    "Insights": "<single line with labels: MISSING TRUTH: ... | CONTRADICTIONS: ... | DEVIATIONS: ... | ADDITIONAL CONTEXT: ... | OVERALL: ...>"
  }}
}}
3. Overall_Score MUST equal round((Factuality + Completeness) / 2).
4. No markdown, no code fences, no extra keys, no extra prose.
5. If uncertain, still produce the exact JSON shape with best-effort scores.

INPUT
Question: {json.dumps(question, ensure_ascii=False)}
Answer: {json.dumps(gold_answer, ensure_ascii=False)}
Agentic_answer: {json.dumps(generated_answer, ensure_ascii=False)}
"""

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        content = (content or "").strip()
        if not content:
            return None

        # Strip markdown code fences if model accidentally emits them.
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content).strip()

        # Direct parse first (supports both arrays and objects).
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        if content.startswith("{"):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        if content.startswith("["):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        # Salvage the first balanced top-level JSON object or array from mixed output.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = content.find(opener)
            if start == -1:
                continue
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(content)):
                ch = content[idx]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = content[start : idx + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        return None

    def _extract_from_reasoning(self, content: str) -> Optional[Dict[str, Any]]:
        content = (content or "").strip()
        # Salvage 1-10 integer scores from free-form text when the model answers
        # semantically but misses exact JSON contract.
        f_pat = [
            r"factuality[^0-9]{0,30}(10|[1-9])",
            r"factuality\s*[:=]\s*(10|[1-9])",
            r"factuality scored\s*(10|[1-9])",
            r"correctness[^0-9]{0,30}(10|[1-9])",
            r"correctness\s*[:=]\s*(10|[1-9])",
            r"truthfulness[^0-9]{0,30}(10|[1-9])",
            r"truthfulness\s*[:=]\s*(10|[1-9])",
            r"faithfulness[^0-9]{0,30}(10|[1-9])",
            r"faithfulness\s*[:=]\s*(10|[1-9])",
            r"groundedness[^0-9]{0,30}(10|[1-9])",
            r"groundedness\s*[:=]\s*(10|[1-9])",
            r"accuracy[^0-9]{0,30}(10|[1-9])",
            r"accuracy\s*[:=]\s*(10|[1-9])",
        ]
        c_pat = [
            r"completeness[^0-9]{0,30}(10|[1-9])",
            r"completeness\s*[:=]\s*(10|[1-9])",
            r"completeness scored\s*(10|[1-9])",
            r"coverage[^0-9]{0,30}(10|[1-9])",
            r"coverage\s*[:=]\s*(10|[1-9])",
            r"thoroughness[^0-9]{0,30}(10|[1-9])",
            r"thoroughness\s*[:=]\s*(10|[1-9])",
            r"recall[^0-9]{0,30}(10|[1-9])",
            r"recall\s*[:=]\s*(10|[1-9])",
            r"relevance[^0-9]{0,30}(10|[1-9])",
            r"relevance\s*[:=]\s*(10|[1-9])",
        ]
        o_pat = [
            r"overall[_\s-]*score[^0-9]{0,30}(10|[1-9])",
            r"overall[_\s-]*score\s*[:=]\s*(10|[1-9])",
            r"overall[^0-9]{0,30}(10|[1-9])",
            r"overall\s*[:=]\s*(10|[1-9])",
            r"final[_\s-]*score[^0-9]{0,30}(10|[1-9])",
            r"final[_\s-]*score\s*[:=]\s*(10|[1-9])",
            r"total[_\s-]*score[^0-9]{0,30}(10|[1-9])",
            r"total[_\s-]*score\s*[:=]\s*(10|[1-9])",
            r"rating[^0-9]{0,30}(10|[1-9])",
            r"rating\s*[:=]\s*(10|[1-9])",
        ]

        def _pick(patterns: List[str]) -> Optional[int]:
            for p in patterns:
                m = re.search(p, content, re.IGNORECASE)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        continue
            return None

        factuality = _pick(f_pat)
        completeness = _pick(c_pat)
        if factuality is not None and completeness is not None:
            overall = _pick(o_pat)
            if overall is None:
                overall = int(round((factuality + completeness) / 2))
            factuality = max(1, min(10, factuality))
            completeness = max(1, min(10, completeness))
            overall = max(1, min(10, overall))
            reason_match = re.search(
                r"reason[^A-Za-z0-9]{0,10}(.+?)(?:insights|$)",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            insights_match = re.search(
                r"insights[^A-Za-z0-9]{0,10}(.+)$",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            return {
                "correctness": factuality / 10.0,
                "completeness": completeness / 10.0,
                "citation_quality": max(0.0, min(1.0, overall / 10.0)),
                "hallucination_penalty": max(0.0, min(1.0, 1.0 - (factuality / 10.0))),
                "format_penalty": 0.0,
                "clarity": max(0.0, min(1.0, overall / 10.0)),
                "final_score": max(0.0, min(1.0, overall / 10.0)),
                "reason": (
                    str(reason_match.group(1)).strip()[:500]
                    if reason_match
                    else "salvaged_from_reasoning_scores"
                ),
                "insights": (
                    str(insights_match.group(1)).strip()[:4000]
                    if insights_match
                    else content[:4000]
                ),
                "factuality_10": factuality,
                "completeness_10": completeness,
                "overall_10": overall,
                "terminal_status": "judge_salvaged_text",
            }

        required = [
            "correctness",
            "completeness",
            "citation_quality",
            "hallucination_penalty",
            "format_penalty",
            "clarity",
            "final_score",
        ]

        found: Dict[str, float] = {}
        reason_segments: List[str] = []

        for key in required:
            patterns = [
                rf'{key}[\s:]+(\d+\.?\d*)',
                rf'"{key}"[\s:]+(\d+\.?\d*)',
                rf'{key}\s*=\s*(\d+\.?\d*)',
            ]
            for pat in patterns:
                m = re.search(pat, content, re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    found[key] = max(0.0, min(1.0, val))
                    break

        reason_match = re.search(
            r'reason["\s:]+["\'](.+?)["\']', content, re.IGNORECASE | re.DOTALL
        )
        if reason_match:
            reason_segments.append(reason_match.group(1).strip())

        if len(found) >= 5:
            for key in required:
                found.setdefault(key, 0.0)
            found["reason"] = " | ".join(reason_segments) if reason_segments else "extracted from reasoning"
            return found

        return None

    def _parse_judge_response(self, content: str) -> Dict[str, Any]:
        data = self._extract_json(content)
        if data is None:
            salvaged = self._extract_from_reasoning(content)
            if salvaged is not None:
                return salvaged
            return self._failed_judge_result("judge_parse_failed:no_parseable_output", content)

        # Accept object or array with first element object.
        if isinstance(data, list):
            if not data or not isinstance(data[0], dict):
                return self._failed_judge_result("judge_parse_failed:invalid_array_shape", content)
            data = data[0]
        if not isinstance(data, dict):
            return self._failed_judge_result("judge_parse_failed:root_not_object", content)

        score = data.get("score")
        if not isinstance(score, dict):
            for nested_key in ("Score", "scores", "Scores"):
                candidate = data.get(nested_key)
                if isinstance(candidate, dict):
                    score = candidate
                    break
            if not isinstance(score, dict):
                score = data

        required = ["Factuality", "Completeness", "Overall_Score", "Reason", "Insights"]
        for key in required:
            if key not in score:
                alias_map = {
                    "Factuality": [
                        "factuality",
                        "correctness",
                        "truthfulness",
                        "faithfulness",
                        "groundedness",
                        "accuracy",
                    ],
                    "Completeness": [
                        "completeness",
                        "coverage",
                        "thoroughness",
                        "recall",
                        "relevance",
                    ],
                    "Overall_Score": [
                        "overall_score",
                        "overall",
                        "overallscore",
                        "final_score",
                        "totalscore",
                        "total_score",
                        "rating",
                    ],
                    "Reason": ["reason", "rationale"],
                    "Insights": ["insights", "analysis"],
                }
                value = None
                for alias in alias_map.get(key, []):
                    if alias in score:
                        value = score[alias]
                        break
                if value is None:
                    salvaged = self._extract_from_reasoning(content)
                    if salvaged is not None:
                        return salvaged
                    return self._failed_judge_result(f"judge_parse_failed:missing_{key}", content)
                score[key] = value

        try:
            factuality = int(score["Factuality"])
            completeness = int(score["Completeness"])
            overall = int(score["Overall_Score"])
        except Exception:
            salvaged = self._extract_from_reasoning(content)
            if salvaged is not None:
                return salvaged
            return self._failed_judge_result("judge_parse_failed:non_integer_scores", content)

        if not (1 <= factuality <= 10 and 1 <= completeness <= 10 and 1 <= overall <= 10):
            salvaged = self._extract_from_reasoning(content)
            if salvaged is not None:
                return salvaged
            return self._failed_judge_result("judge_parse_failed:score_out_of_range", content)

        expected_overall = int(round((factuality + completeness) / 2.0))
        if overall != expected_overall:
            overall = expected_overall

        return {
            "correctness": factuality / 10.0,
            "completeness": completeness / 10.0,
            "citation_quality": overall / 10.0,
            "hallucination_penalty": max(0.0, min(1.0, 1.0 - (factuality / 10.0))),
            "format_penalty": 0.0,
            "clarity": overall / 10.0,
            "final_score": overall / 10.0,
            "reason": str(score.get("Reason", ""))[:500],
            "insights": str(score.get("Insights", ""))[:4000],
            "factuality_10": factuality,
            "completeness_10": completeness,
            "overall_10": overall,
            "terminal_status": "judge_structured_ok",
        }

    def _validate_and_normalize(
        self, data: Dict[str, Any], raw: str
    ) -> Dict[str, Any]:
        required = [
            "correctness",
            "completeness",
            "citation_quality",
            "hallucination_penalty",
            "format_penalty",
            "clarity",
            "final_score",
        ]

        for key in required:
            if key not in data:
                return self._failed_judge_result(f"missing_{key}", raw)

        for key in required:
            try:
                data[key] = float(data[key])
            except (TypeError, ValueError):
                return self._failed_judge_result(f"invalid_float_{key}", raw)
            data[key] = max(0.0, min(1.0, data[key]))

        recomputed = (
            0.45 * data["correctness"]
            + 0.25 * data["completeness"]
            + 0.10 * data["citation_quality"]
            + 0.10 * data["clarity"]
            - 0.30 * data["hallucination_penalty"]
            - 0.15 * data["format_penalty"]
        )
        data["judge_final_score"] = max(0.0, min(1.0, data["final_score"]))
        data["final_score"] = max(0.0, min(1.0, recomputed))
        data["reason"] = str(data.get("reason", ""))[:500]
        return data

    def _failed_judge_result(self, error: str, raw: str) -> Dict[str, Any]:
        return {
            "correctness": None,
            "completeness": None,
            "citation_quality": None,
            "hallucination_penalty": None,
            "format_penalty": None,
            "clarity": None,
            "final_score": None,
            "reason": error,
            "raw_response": raw[:1000],
            "judge_raw_response": raw[:8000],
            "terminal_status": "judge_parse_failed",
        }

    def evaluate_details(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> Dict[str, Any]:
        key = self._cache_key(question, gold_answer, generated_answer)

        if key in self.cache:
            return self.cache[key]

        prompt = self._build_prompt(question, gold_answer, generated_answer)

        response = None
        last_error: Optional[Exception] = None
        attempts = 0
        had_retry = False
        request_started = time.time()
        last_error_type = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                attempts += 1
                request: Dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": True
                        }
                    },
                    "response_format": {"type": "json_object"},
                }
                if self.temperature is not None:
                    request["temperature"] = self.temperature
                if self.max_tokens is not None:
                    request["max_tokens"] = self.max_tokens
                response = self.client.chat.completions.create(**request)
                break
            except Exception as exc:
                last_error = exc
                last_error_type = type(exc).__name__
                if attempt < self.max_retries:
                    had_retry = True
                    time.sleep(0.75 * attempt)

        if response is None:
            logger.warning("Judge request failed after retries: %s", last_error)
            result = self._failed_judge_result(
                f"judge_request_failed:{str(last_error) if last_error else 'unknown_error'}",
                str(last_error) if last_error else "unknown error",
            )
            result["terminal_status"] = "judge_request_failed"
            result["judge_attempts"] = attempts
            result["judge_retried"] = had_retry
            result["judge_latency_ms"] = int((time.time() - request_started) * 1000)
            result["judge_error_type"] = last_error_type
            self.cache[key] = result
            if self.cache_path:
                self._save_cache()
            return result

        content = response.choices[0].message.content or ""
        result = self._parse_judge_response(content)
        result["judge_raw_response"] = content[:8000]
        result["judge_attempts"] = attempts
        result["judge_retried"] = had_retry
        result["judge_latency_ms"] = int((time.time() - request_started) * 1000)
        result["judge_error_type"] = last_error_type
        if result.get("final_score") is not None:
            result["final_score"] = max(0.0, min(1.0, float(result["final_score"])))

        self.cache[key] = result
        if self.cache_path:
            self._save_cache()

        return result

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def evaluate(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> float:
        return self.evaluate_details(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )["final_score"]

    def evaluate_reward(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> float:
        score = self.evaluate(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )
        if self.reward_mode == "minus_one_to_one":
            return 2.0 * score - 1.0
        return score
