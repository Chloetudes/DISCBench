# -*- coding: utf-8 -*-
"""
clients/openai_client.py - 统一客户端（修复版 - 分离 reasoning 和 content）

默认使用流式（SSE）接收回复：连接保持活跃，按 chunk 读取，避免长回复整包等待导致 read timeout。
可通过 chat(..., stream=False) 关闭。
"""

import json
import requests
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class OAIClient:
    """统一的OpenAI兼容客户端"""

    _STREAM_PROTOCOLS = frozenset({"openai", "dashscope"})

    def __init__(self, base_url: str, api_key: str,
                 protocol: str = "openai",
                 auth_header: str = "Authorization",
                 auth_prefix: str = "Bearer",
                 extra_headers: Optional[Dict[str, str]] = None,
                 timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.protocol = protocol
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

    def _build_headers(self, *, sse: bool = False) -> Dict[str, str]:
        if self.auth_prefix:
            auth_value = f"{self.auth_prefix} {self.api_key}"
        else:
            auth_value = self.api_key

        headers = {
            "Content-Type": "application/json",
            self.auth_header: auth_value,
        }
        if sse and self.protocol == "dashscope":
            headers["X-DashScope-SSE"] = "enable"
        headers.update(self.extra_headers)
        return headers

    def _build_openai_payload(self, model: str, messages: List[Dict],
                              temperature: float, stream: bool = False,
                              **kwargs) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            payload.setdefault("stream_options", {"include_usage": False})
        payload.update(kwargs)
        return payload

    def _build_vertex_payload(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"[System]: {msg['content']}"}]
                })
            else:
                content = msg["content"]
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if item.get("type") == "text":
                            parts.append({"text": item["text"]})
                    contents.append({
                        "role": msg["role"],
                        "parts": parts
                    })
                else:
                    contents.append({
                        "role": msg["role"],
                        "parts": [{"text": content}]
                    })

        payload = {"contents": contents}
        payload.update(kwargs)
        return payload

    def _build_openai_responses_payload(self, model: str, messages: List[Dict],
                                        temperature: float, stream: bool = False,
                                        **kwargs) -> Dict[str, Any]:
        payload = {
            "model": model,
            "input": messages,
            "temperature": temperature,
            "stream": stream,
        }
        payload.update(kwargs)
        return payload

    def _build_dashscope_payload(self, model: str, messages: List[Dict],
                                 temperature: float, stream: bool = False,
                                 **kwargs) -> Dict[str, Any]:
        formatted_messages = []
        for msg in messages:
            content = msg["content"]
            formatted_msg = {
                "role": msg["role"],
                "content": content if isinstance(content, str) else str(content)
            }
            formatted_messages.append(formatted_msg)

        payload = {
            "model": model,
            "input": {
                "messages": formatted_messages
            },
            "parameters": {
                "result_format": "message",
                "temperature": temperature,
            }
        }
        if stream:
            payload["parameters"]["incremental_output"] = True
        if "max_tokens" in kwargs:
            payload["parameters"]["max_tokens"] = kwargs["max_tokens"]

        if kwargs.get("enable_thinking", True):
            payload["parameters"]["enable_thinking"] = True

        if kwargs.get("enable_search", False):
            payload["parameters"]["enable_search"] = True
            payload["parameters"]["search_options"] = {
                "search_strategy": kwargs.get("search_strategy", "agent_max"),
                "enable_source": kwargs.get("enable_source", True)
            }

        if kwargs.get("enable_code_interpreter", False):
            payload["parameters"]["enable_code_interpreter"] = True

        return payload

    def _build_url(self, model: str = None) -> str:
        if self.protocol == "vertex":
            if not model:
                raise ValueError("Vertex协议需要指定model")
            return f"{self.base_url}/models/{model}:generateContent"
        elif self.protocol == "openai_responses":
            return f"{self.base_url}"
        elif self.protocol == "dashscope":
            return self.base_url
        else:
            return f"{self.base_url}/chat/completions"

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        """content 可能是 str，或 Anthropic/OpenAI 的 [{type, text}, ...] 列表。"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict):
                    t = blk.get("text")
                    if t is None:
                        t = blk.get("content")
                    if t is not None:
                        parts.append(str(t))
                elif blk is not None:
                    parts.append(str(blk))
            return "\n".join(parts)
        return str(content)

    def _supports_stream(self) -> bool:
        return self.protocol in self._STREAM_PROTOCOLS

    @staticmethod
    def _request_timeout_tuple() -> Tuple[int, int]:
        """(connect_timeout, read_timeout_between_chunks)"""
        return (30, 600)

    @staticmethod
    def _iter_sse_data_lines(response: requests.Response) -> Iterator[Dict[str, Any]]:
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    break
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    def _accumulate_openai_stream_chunk(
        self,
        chunk: Dict[str, Any],
        content_parts: List[str],
        reasoning_parts: List[str],
        finish_reason: Optional[str],
    ) -> Optional[str]:
        if chunk.get("error"):
            err = chunk["error"]
            if isinstance(err, dict):
                msg = err.get("message") or str(err)
            else:
                msg = str(err)
            raise RuntimeError(f"流式错误: {msg}")

        choices = chunk.get("choices") or []
        if not choices:
            # DashScope 原生 SSE 可能直接在 output.choices
            output = chunk.get("output") or {}
            choices = output.get("choices") or []

        for choice in choices:
            fr = choice.get("finish_reason")
            if fr and fr not in ("null", "None"):
                finish_reason = fr

            delta = choice.get("delta") or {}
            message = choice.get("message") or {}

            for key in ("content",):
                piece = delta.get(key)
                if piece is None and message:
                    piece = message.get(key)
                if piece:
                    content_parts.append(self._message_content_to_text(piece))

            for key in ("reasoning_content", "reasoning"):
                piece = delta.get(key)
                if piece is None and message:
                    piece = message.get(key)
                if piece:
                    reasoning_parts.append(str(piece))

        return finish_reason

    def _chat_stream_with_meta(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason: Optional[str] = None

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=self._request_timeout_tuple(),
        )
        try:
            response.raise_for_status()
            for chunk in self._iter_sse_data_lines(response):
                finish_reason = self._accumulate_openai_stream_chunk(
                    chunk, content_parts, reasoning_parts, finish_reason
                )
        finally:
            response.close()

        text = "".join(content_parts)
        reasoning = "".join(reasoning_parts) or None
        return text, finish_reason, reasoning

    def _parse_openai_response(self, result: Dict) -> str:
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "")
            text = self._message_content_to_text(content)
            return text if text else str(result)
        return str(result)

    def _parse_vertex_response(self, result: Dict) -> str:
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                return "".join(part.get("text", "") for part in parts)
        return str(result)

    def _parse_openai_responses_response(self, result: Dict) -> str:
        if "output" in result:
            return result["output"]
        return str(result)

    def _parse_dashscope_response(self, result: Dict,
                                  return_reasoning: bool = False) -> Union[str, Tuple[str, str]]:
        try:
            output = result.get("output", {})
            choices = output.get("choices", [])

            if not choices:
                if "text" in output:
                    return (output["text"], "") if return_reasoning else output["text"]
                return (str(result), "") if return_reasoning else str(result)

            message = choices[0].get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")

            if return_reasoning:
                return content, reasoning
            else:
                return content

        except Exception as e:
            error_msg = f"<error: 解析失败 - {str(e)}>"
            return (error_msg, "") if return_reasoning else error_msg

    def _build_payload_and_url(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        stream: bool,
        kwargs: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        url = self._build_url(model)
        if self.protocol == "dashscope":
            payload = self._build_dashscope_payload(
                model, messages, temperature, stream=stream, **kwargs
            )
        elif self.protocol == "vertex":
            payload = self._build_vertex_payload(messages, **kwargs)
        elif self.protocol == "openai_responses":
            payload = self._build_openai_responses_payload(
                model, messages, temperature, stream=stream, **kwargs
            )
        else:
            payload = self._build_openai_payload(
                model, messages, temperature, stream=stream, **kwargs
            )
        return url, payload

    def chat(self, model: str, messages: List[Dict],
             temperature: float = 0.7, **kwargs) -> str:
        use_stream = kwargs.pop("stream", True)
        if use_stream and self._supports_stream():
            text, _, _ = self.chat_with_meta(
                model, messages, temperature=temperature, stream=True, **kwargs
            )
            return text

        url = self._build_url(model)
        headers = self._build_headers()

        if self.protocol == "dashscope":
            payload = self._build_dashscope_payload(
                model, messages, temperature, stream=False, **kwargs
            )
        elif self.protocol == "vertex":
            payload = self._build_vertex_payload(messages, **kwargs)
        elif self.protocol == "openai_responses":
            payload = self._build_openai_responses_payload(
                model, messages, temperature, **kwargs
            )
        else:
            payload = self._build_openai_payload(
                model, messages, temperature, **kwargs
            )

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            if self.protocol == "dashscope":
                return self._parse_dashscope_response(result, return_reasoning=False)
            elif self.protocol == "vertex":
                return self._parse_vertex_response(result)
            elif self.protocol == "openai_responses":
                return self._parse_openai_responses_response(result)
            else:
                return self._parse_openai_response(result)

        except requests.exceptions.Timeout:
            raise TimeoutError(f"请求超时 ({self.timeout}s)")
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get("message", str(e))
            except Exception:
                error_msg = f"{str(e)}\n{e.response.text[:200]}"
            raise RuntimeError(f"请求失败: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"请求失败: {str(e)}")

    def dashscope_raw_request(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        调用 DashScope 文本生成接口并返回完整 JSON（用于解析 search_info 等）。
        需在构造 OAIClient 时 protocol='dashscope'，且 base_url 为原生 generation 端点。
        """
        if self.protocol != "dashscope":
            raise ValueError("dashscope_raw_request 仅适用于 protocol='dashscope'")
        url = self._build_url(model)
        headers = self._build_headers()
        payload = self._build_dashscope_payload(
            model, messages, temperature, stream=False, **kwargs
        )
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def chat_with_meta(self, model: str, messages: List[Dict],
                       temperature: float = 0.7, **kwargs) -> Tuple[str, Optional[str], Optional[str]]:
        """发送聊天请求并返回元数据（默认流式 SSE）。

        Returns:
            (content, finish_reason, reasoning)
        """
        use_stream = kwargs.pop("stream", True)
        url, payload = self._build_payload_and_url(
            model, messages, temperature, stream=use_stream and self._supports_stream(), kwargs=kwargs
        )

        if use_stream and self._supports_stream():
            headers = self._build_headers(sse=True)
            try:
                text, finish_reason, reasoning = self._chat_stream_with_meta(url, headers, payload)
                return text or "", finish_reason, reasoning
            except requests.exceptions.Timeout:
                raise TimeoutError(f"流式请求超时 (chunk read>{self._request_timeout_tuple()[1]}s)")
            except requests.exceptions.HTTPError as e:
                try:
                    error_detail = e.response.json()
                    error_msg = error_detail.get("message", str(e))
                except Exception:
                    error_msg = f"{str(e)}\n{e.response.text[:200]}"
                raise RuntimeError(f"请求失败: {error_msg}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"请求失败: {str(e)}")

        headers = self._build_headers()
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            text = None
            finish_reason = None
            reasoning = None

            if self.protocol == "vertex":
                text = self._parse_vertex_response(result)
                if "candidates" in result and len(result["candidates"]) > 0:
                    finish_reason = result["candidates"][0].get("finishReason")
            elif self.protocol == "openai_responses":
                text = self._parse_openai_responses_response(result)
                finish_reason = result.get("finish_reason")
            elif self.protocol == "dashscope":
                text, reasoning = self._parse_dashscope_response(result, return_reasoning=True)
                output = result.get("output", {})
                choices = output.get("choices", [])
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason == 'null':
                        finish_reason = None
            else:
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    text = self._message_content_to_text(
                        choice.get("message", {}).get("content", "")
                    )
                    finish_reason = choice.get("finish_reason")

            return text or str(result), finish_reason, reasoning

        except requests.exceptions.Timeout:
            raise TimeoutError(f"请求超时 ({self.timeout}s)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"请求失败: {str(e)}")
