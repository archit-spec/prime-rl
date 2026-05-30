import asyncio
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import logging


logger = logging.getLogger("xyne-env")


@dataclass
class Episode:
    question: str
    gold_answer: str = ""
    question_id: str = ""
    prompt_group_id: str = ""
    rollout_index: int = 0
    final_answer: str = ""
    reward: float = 0.0
    advantage: float = 0.0
    raw_response: str = ""
    raw_assistant_tail: str = ""
    success: bool = False
    error: str = ""
    model: str = ""
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    conversation_id: str = ""
    turn_id: str = ""
    run_id: str = ""
    episode_id: str = ""
    final_status: str = "unknown"
    extraction_status: str = "unknown"
    extraction_length: int = 0
    leaked_tool_or_thinking: bool = False
    tool_uses: int = 0
    tool_results: int = 0
    retrieved_hits: int = 0
    events: int = 0
    text_committed: int = 0
    thinking_delta: int = 0
    turns: int = 0
    latency_sec: float = 0.0


class XyneEnvironment:
    def __init__(
        self,
        base_url: str = "http://localhost:23000",
        timeout: int = 300,
        max_episode_duration: int = 600,
        stream_heartbeat_sec: int = 30,
        model: str = "private-large",
        reasoning: bool = True,
        websearch: bool = False,
        deep_research: bool = False,
        agentic: bool = True,
        auth_cookie: Optional[str] = None,
        cookie_env_path: Optional[str] = None,
        max_retries: int = 3,
        agent_id: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_episode_duration = max(30, int(max_episode_duration))
        self.stream_heartbeat_sec = max(5, int(stream_heartbeat_sec))
        self.model = model
        self.reasoning = reasoning
        self.websearch = websearch
        self.deep_research = deep_research
        self.agentic = agentic
        self.max_retries = max_retries
        self.agent_id = (agent_id or "").strip()
        self.auth_cookie = (
            auth_cookie
            or os.environ.get("XYNE_AUTH_COOKIE")
            or os.environ.get("TEST_API_COOKIES")
            or self._read_cookie_from_env_file(cookie_env_path)
        )

    @staticmethod
    def _read_cookie_from_env_file(cookie_env_path: Optional[str]) -> str:
        candidates = [
            cookie_env_path,
            os.environ.get("XYNE_COOKIE_FILE"),
            os.environ.get("XYNE_SERVER_ENV"),
            "/data/RL-Training/RL-Training/xyne/server/.env",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if not path.exists():
                continue
            text = path.read_text()
            match = re.search(
                r"^TEST_API_COOKIES\s*=\s*[\"']?(.*?)[\"']?\s*$",
                text,
                re.MULTILINE,
            )
            if match:
                return match.group(1).strip()
            raw = text.strip()
            if raw.startswith("access-token="):
                return raw
        return ""

    def _headers(self, stream: bool = False) -> Dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream" if stream else "application/json",
            "User-Agent": "XyneGRPORollout/1.0",
            "Content-Type": "application/json",
        }
        if self.auth_cookie:
            cookie = self.auth_cookie.strip()
            headers["Cookie"] = cookie if cookie.startswith("access-token=") else f"access-token={cookie}"
        return headers

    async def healthcheck(self) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v2/me",
                timeout=aiohttp.ClientTimeout(total=10),
                headers=self._headers(),
            ) as resp:
                text = await resp.text()
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"raw": text}
                payload["status_code"] = resp.status
                return payload

    async def run_episode(
        self,
        question: str,
        gold_answer: str = "",
        question_id: str = "",
        prompt_group_id: str = "",
        rollout_index: int = 0,
        event_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Episode:
        effective_group_id = prompt_group_id or question_id
        episode = Episode(
            question=question,
            gold_answer=gold_answer,
            question_id=effective_group_id,
            prompt_group_id=effective_group_id,
            rollout_index=int(rollout_index),
            model=self.model,
        )
        attempt_started = time.time()
        last_error = ""
        collected_events: List[Dict[str, Any]] = []
        collected_text: List[str] = []
        trajectory: List[Dict[str, Any]] = []

        for attempt in range(1, self.max_retries + 2):
            try:
                conversation_id = await self._create_conversation()
                if event_logger:
                    event_logger(
                        {
                            "type": "rollout_start",
                            "ts": time.time(),
                            "question_id": question_id,
                            "prompt_group_id": effective_group_id,
                            "rollout_index": int(rollout_index),
                            "question": question,
                            "conversation_id": conversation_id,
                            "attempt": attempt,
                            "model": self.model,
                        }
                    )
                send_payload = await self._send_message(conversation_id, question)
                status, stream_stats = await self._listen_stream(
                    conversation_id,
                    collected_text,
                    trajectory,
                    collected_events,
                    question_id=effective_group_id,
                    prompt_group_id=effective_group_id,
                    rollout_index=int(rollout_index),
                    question=question,
                    event_logger=event_logger,
                )

                episode.raw_response = json.dumps(collected_events)
                episode.trajectory = trajectory
                episode.conversation_id = conversation_id
                episode.turn_id = stream_stats.get("turn_id") or send_payload.get("turn_id", "")
                episode.run_id = stream_stats.get("run_id") or send_payload.get("run_id", "")
                episode.episode_id = ":".join(part for part in [conversation_id, episode.turn_id or episode.run_id] if part)
                episode.final_status = stream_stats.get("final_status", status or "unknown")
                episode.tool_uses = int(stream_stats.get("tool_uses", 0))
                episode.tool_results = int(stream_stats.get("tool_results", 0))
                episode.retrieved_hits = int(stream_stats.get("retrieved_hits", 0))
                episode.events = int(stream_stats.get("events", 0))
                episode.text_committed = int(stream_stats.get("text_committed", 0))
                episode.thinking_delta = int(stream_stats.get("thinking_delta", 0))
                episode.turns = int(stream_stats.get("turns", 0))

                if status == "completed":
                    extraction = await self._fetch_final_answer_payload(conversation_id)
                    episode.raw_assistant_tail = str(extraction.get("raw_assistant_tail", ""))
                    episode.final_answer = str(extraction.get("final_answer", ""))
                    episode.extraction_status = str(extraction.get("extraction_status", "unknown"))
                    episode.extraction_length = int(extraction.get("extraction_length", 0) or 0)
                    episode.leaked_tool_or_thinking = bool(extraction.get("leaked_tool_or_thinking", False))
                    if not episode.final_answer.strip():
                        episode.success = False
                        episode.error = "final_answer_missing"
                    else:
                        episode.success = True
                    episode.latency_sec = round(time.time() - attempt_started, 3)
                    self._emit_event(event_logger, episode)
                    return episode

                if status == "errored":
                    error_msg = "turn_errored"
                    for ev in collected_events:
                        if ev.get("event") == "turn_ended" and ev.get("data", {}).get("status") == "errored":
                            error_msg = ev["data"].get("error", "turn_errored")
                            break
                    episode.error = error_msg
                else:
                    final_status = stream_stats.get("final_status", "unknown")
                    episode.error = f"stream_timeout_or_unknown_status:{final_status}"

                episode.final_answer = "".join(collected_text)
                episode.success = False
                episode.latency_sec = round(time.time() - attempt_started, 3)
                self._emit_event(event_logger, episode)
                return episode

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                collected_events.append(
                    {
                        "event": "exception",
                        "error": last_error,
                        "traceback": traceback.format_exc(),
                    }
                )
                if attempt <= self.max_retries:
                    await asyncio.sleep(min(2**attempt, 8))
                else:
                    break

        episode.error = last_error
        episode.raw_response = json.dumps(collected_events)
        episode.trajectory = trajectory
        episode.final_answer = "".join(collected_text)
        episode.success = False
        episode.latency_sec = round(time.time() - attempt_started, 3)
        self._emit_event(event_logger, episode)
        return episode

    @staticmethod
    def _emit_event(event_logger: Optional[Callable[[Dict[str, Any]], None]], episode: Episode) -> None:
        if not event_logger:
            return
        event_logger(
            {
                "type": "rollout_episode",
                "question_id": episode.question_id,
                "prompt_group_id": episode.prompt_group_id,
                "rollout_index": episode.rollout_index,
                "question": episode.question,
                "episode_id": episode.episode_id,
                "conversation_id": episode.conversation_id,
                "turn_id": episode.turn_id,
                "run_id": episode.run_id,
                "final_status": episode.final_status,
                "tool_uses": episode.tool_uses,
                "tool_results": episode.tool_results,
                "retrieved_hits": episode.retrieved_hits,
                "events": episode.events,
                "text_committed": episode.text_committed,
                "thinking_delta": episode.thinking_delta,
                "turns": episode.turns,
                "raw_assistant_tail": episode.raw_assistant_tail,
                "extraction_status": episode.extraction_status,
                "extraction_length": episode.extraction_length,
                "leaked_tool_or_thinking": episode.leaked_tool_or_thinking,
                "final_answer_present": bool(episode.final_answer.strip()),
                "final_answer": episode.final_answer,
                "success": episode.success,
                "error": episode.error,
                "latency_sec": episode.latency_sec,
            }
        )

    async def _create_conversation(self) -> str:
        url = f"{self.base_url}/v2/chat/conversations"
        body = {"title": "RL Training"}
        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=self._headers(), json=body) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Create conversation failed: {resp.status} {text[:1000]}")
                data = await resp.json()
                conv_id = data.get("id")
                if not conv_id:
                    raise RuntimeError(f"Missing conversation id in response: {data}")
                return conv_id

    async def _send_message(self, conversation_id: str, text: str) -> Dict[str, str]:
        url = f"{self.base_url}/v2/chat/conversations/{conversation_id}/messages"
        body: Dict[str, Any] = {"text": text, "model": self.model}
        if self.agent_id:
            body["agentId"] = self.agent_id
        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=self._headers(), json=body) as resp:
                body_text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"Send message failed: {resp.status} {body_text[:1000]}")
                try:
                    payload = json.loads(body_text)
                except json.JSONDecodeError:
                    payload = {}
                return {
                    "turn_id": payload.get("turn", {}).get("id", ""),
                    "run_id": payload.get("assistantMessage", {}).get("runId", ""),
                }

    async def _listen_stream(
        self,
        conversation_id: str,
        collected_text: List[str],
        trajectory: List[Dict[str, Any]],
        collected_events: List[Dict[str, Any]],
        question_id: str = "",
        prompt_group_id: str = "",
        rollout_index: int = 0,
        question: str = "",
        event_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.base_url}/v2/chat/conversations/{conversation_id}/stream"
        timeout = aiohttp.ClientTimeout(total=self.timeout, sock_read=120)
        stats: Dict[str, Any] = {
            "tool_uses": 0,
            "tool_results": 0,
            "retrieved_hits": 0,
            "events": 0,
            "text_committed": 0,
            "thinking_delta": 0,
            "turn_id": "",
            "run_id": "",
            "final_status": "unknown",
            "turns": 0,
        }
        stream_started = time.time()
        next_heartbeat = stream_started + self.stream_heartbeat_sec

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=self._headers(stream=True)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"SSE stream failed: {resp.status} {text[:1000]}")

                current_event_name: Optional[str] = None
                current_data_lines: List[str] = []

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

                    if line.startswith("event: "):
                        current_event_name = line[7:]
                    elif line.startswith("data: "):
                        current_data_lines.append(line[6:])
                    elif line == "":
                        if current_event_name and current_data_lines:
                            data_str = "\n".join(current_data_lines)
                            try:
                                data = json.loads(data_str) if data_str.strip() else {}
                            except json.JSONDecodeError:
                                data = {"raw": data_str}

                            event_obj = {"event": current_event_name, "data": data}
                            collected_events.append(event_obj)
                            stats["events"] += 1

                            if current_event_name == "text_committed":
                                stats["text_committed"] += 1
                                txt = data.get("text", "")
                                if txt:
                                    collected_text.append(txt)
                            elif current_event_name == "thinking_delta":
                                stats["thinking_delta"] += 1
                            elif current_event_name == "turn_started":
                                stats["turns"] += 1
                                stats["turn_id"] = data.get("turnId", stats["turn_id"])
                            elif current_event_name == "run_started":
                                stats["run_id"] = data.get("runId", stats["run_id"])
                            elif current_event_name == "block_appended":
                                block = data.get("block", {})
                                kind = block.get("kind")
                                if kind == "tool_use":
                                    stats["tool_uses"] += 1
                                    trajectory.append(
                                        {
                                            "type": "tool_call",
                                            "tool_name": block.get("toolName", ""),
                                            "args": block.get("args", {}),
                                        }
                                    )
                                elif kind == "tool_result":
                                    stats["tool_results"] += 1
                                    trajectory.append(
                                        {
                                            "type": "tool_result",
                                            "tool_call_id": block.get("toolCallId", ""),
                                            "output": block.get("output", ""),
                                            "is_error": block.get("isError", False),
                                        }
                                    )
                                    stats["retrieved_hits"] += self._count_retrieved_hits(block.get("output"))
                            elif current_event_name == "turn_ended":
                                status = data.get("status", "")
                                stats["final_status"] = status or "unknown"
                                return status, stats

                        now = time.time()
                        elapsed = now - stream_started
                        if elapsed >= self.max_episode_duration:
                            stats["final_status"] = "max_episode_duration_exceeded"
                            logger.warning(
                                "episode timeout qid=%s conv=%s elapsed=%.1fs max=%ss events=%d tools=%d",
                                question_id or "unknown",
                                conversation_id,
                                elapsed,
                                self.max_episode_duration,
                                stats["events"],
                                stats["tool_uses"],
                            )
                            return "timeout", stats

                        if now >= next_heartbeat:
                            logger.info(
                                "episode heartbeat qid=%s conv=%s elapsed=%.1fs events=%d tools=%d hits=%d thinking=%d",
                                question_id or "unknown",
                                conversation_id,
                                elapsed,
                                stats["events"],
                                stats["tool_uses"],
                                stats["retrieved_hits"],
                                stats["thinking_delta"],
                            )
                            if event_logger:
                                event_logger(
                                    {
                                        "type": "rollout_progress",
                                        "ts": now,
                                        "question_id": question_id,
                                        "prompt_group_id": prompt_group_id or question_id,
                                        "rollout_index": int(rollout_index),
                                        "question": question,
                                        "conversation_id": conversation_id,
                                        "elapsed_sec": round(elapsed, 3),
                                        "events": stats["events"],
                                        "tool_uses": stats["tool_uses"],
                                        "tool_results": stats["tool_results"],
                                        "retrieved_hits": stats["retrieved_hits"],
                                        "thinking_delta": stats["thinking_delta"],
                                        "text_committed": stats["text_committed"],
                                        "turns": stats["turns"],
                                    }
                                )
                            next_heartbeat = now + self.stream_heartbeat_sec

                        current_event_name = None
                        current_data_lines = []

                return "timeout", stats

    @staticmethod
    def _count_retrieved_hits(output: Any) -> int:
        text = ""
        if isinstance(output, dict):
            content = output.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text += item.get("text", "")
        elif isinstance(output, str):
            text = output
        return len(re.findall(r"<hit rank=", text))

    @staticmethod
    def _looks_like_tool_or_thinking_leak(text: str) -> bool:
        return bool(
            re.search(
                r"(tool_use|tool_result|thinking_|<hit rank=|\{\"type\":|phase=final_answer)",
                text or "",
                flags=re.IGNORECASE,
            )
        )

    async def _fetch_final_answer_payload(self, conversation_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v2/chat/conversations/{conversation_id}/messages?page=first&limit=200"
        timeout = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=self._headers()) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Fetch messages failed: {resp.status} {text[:1000]}")
                data = await resp.json()

        items = data.get("items", [])
        assistant_messages = [msg for msg in items if msg.get("role") == "assistant"]
        if not assistant_messages:
            return {
                "raw_assistant_tail": "",
                "final_answer": "",
                "extraction_status": "missing_assistant_message",
                "extraction_length": 0,
                "leaked_tool_or_thinking": False,
            }

        last_assistant = max(assistant_messages, key=lambda m: m.get("ordinal", 0))
        blocks = last_assistant.get("blocks", [])
        text_parts = []
        for block in blocks:
            if block.get("kind") == "text":
                text_parts.append(block.get("text", ""))
        raw_assistant_tail = "".join(text_parts)
        final_answer = self.clean_answer(raw_assistant_tail)
        extraction_status = "ok"
        if not blocks:
            extraction_status = "assistant_blocks_missing"
        elif not text_parts:
            extraction_status = "assistant_text_blocks_missing"
        elif not final_answer.strip():
            extraction_status = "assistant_text_empty_after_clean"
        return {
            "raw_assistant_tail": raw_assistant_tail,
            "final_answer": final_answer,
            "extraction_status": extraction_status,
            "extraction_length": len(final_answer),
            "leaked_tool_or_thinking": self._looks_like_tool_or_thinking_leak(raw_assistant_tail),
        }

    @classmethod
    def parse_agentic_response(cls, text: str) -> str:
        if not text:
            return ""

        json_objects = cls._extract_json_objects(text)

        for obj in json_objects:
            if (
                obj.get("type") == "tool_call_end"
                and obj.get("data", {}).get("tool_name") == "synthesize_final_answer"
                and obj.get("data", {}).get("final_output")
            ):
                return cls.clean_answer(str(obj["data"]["final_output"]))

        for obj in json_objects:
            if obj.get("type") == "final_output":
                output = obj.get("data", {}).get("output")
                if isinstance(output, str):
                    return cls.clean_answer(output)

        for obj in json_objects:
            outcome = obj.get("data", {}).get("outcome", {})
            if obj.get("type") == "run_end" and outcome.get("output"):
                return cls.clean_answer(str(outcome["output"]))

        assistant_parts: List[str] = []
        for obj in json_objects:
            message = obj.get("data", {}).get("message", {})
            content = message.get("content")
            if (
                obj.get("type") == "assistant_message"
                and isinstance(content, str)
                and not message.get("tool_calls")
            ):
                assistant_parts.append(content)
        if assistant_parts:
            return cls.clean_answer("".join(assistant_parts))

        after_thinking = text
        thinking_end = after_thinking.rfind('{"type":"thinking_end"')
        if thinking_end != -1:
            end_brace = after_thinking.find("}", thinking_end)
            if end_brace != -1:
                after_thinking = after_thinking[end_brace + 1 :]
        synthesis_start = after_thinking.find('{"type":"synthesis_completed"')
        if synthesis_start != -1:
            after_thinking = after_thinking[:synthesis_start]
        if after_thinking.strip() and after_thinking != text:
            return cls.clean_answer(after_thinking)

        return cls.clean_answer(text)

    @staticmethod
    def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        depth = 0
        start: Optional[int] = None
        in_string = False
        escape = False

        for idx, char in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    start = None

        return objects

    @staticmethod
    def clean_answer(answer: str) -> str:
        cleaned = answer or ""
        cleaned = re.sub(
            r'^\s*\{"contextChunks":\[.*?\],"citationMap":\{.*?\}\}\s*',
            "",
            cleaned,
            flags=re.DOTALL,
        )
        cleaned = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", cleaned)
        cleaned = cleaned.replace("\\n", "\n").replace('\\"', '"')
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip().strip("\"'")

    def compute_reward(self, episode: Episode, judge) -> float:
        if not episode.final_answer:
            return 0.0
        return judge.evaluate(
            question=episode.question,
            gold_answer=episode.gold_answer,
            generated_answer=episode.final_answer,
        )
