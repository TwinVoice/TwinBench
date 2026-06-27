"""Tiny OpenAI-compatible mock server for local smoke tests.

It implements only the chat completions endpoint used by TwinBench. The server
does not evaluate model quality; it only verifies that the pipeline can call an
OpenAI-compatible endpoint, parse responses, and write result files.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _message_content(prompt: str) -> str:
    if "generated_content" in prompt:
        return json.dumps({"generated_content": "Mock persona reply."})
    if "final_score" in prompt:
        return json.dumps(
            {
                "analysis": {
                    "opinion_consistency": {
                        "is_consistent": True,
                        "justification": "Mock response for smoke testing.",
                    },
                    "logical_factual_fidelity": {
                        "is_faithful": True,
                        "justification": "Mock response for smoke testing.",
                    },
                    "stylistic_similarity": {
                        "similarity_level": "Medium",
                        "justification": "Mock response for smoke testing.",
                    },
                },
                "final_score": "3",
                "final_justification": "Mock score for smoke testing.",
            }
        )
    if "Output exactly one letter" in prompt:
        return "A"
    return json.dumps({"choice": "A"})


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}

        messages = payload.get("messages") or []
        prompt = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        content = _message_content(prompt)

        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        raw = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mock OpenAI server listening on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
