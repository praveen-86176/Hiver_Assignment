#!/usr/bin/env python3
"""
Web Server for Apple Support AI Agent Dashboard.
Provides a REST API and serves the modern interactive web application.
"""

import json
import os
import sys
import time
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Add root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.agent.pipeline import SupportAgentPipeline
from src.baselines.baselines import TrivialBaseline, KeywordBaseline

# Global pipeline instance (loaded lazily or on start)
pipeline = None
trivial_baseline = TrivialBaseline()
keyword_baseline = KeywordBaseline()

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("[Server] Initializing SupportAgentPipeline...")
        pipeline = SupportAgentPipeline()
    return pipeline


class AgentDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "web"), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(PROJECT_ROOT / "web" / "index.html", "rb") as f:
                self.wfile.write(f.read())
            return

        elif parsed.path == "/api/taxonomy":
            self.send_json_response(self.get_taxonomy())
            return

        elif parsed.path == "/api/golden_set":
            self.send_json_response(self.get_golden_set())
            return

        elif parsed.path == "/api/stats":
            self.send_json_response(self.get_stats())
            return

        elif parsed.path == "/api/eval_report":
            self.send_json_response(self.get_eval_report())
            return

        # Serve static files from web/
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/run_agent":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                message = data.get("message", "").strip()
                system_type = data.get("system", "agent")

                if not message:
                    self.send_json_response({"error": "Message cannot be empty"}, status=400)
                    return

                t0 = time.time()
                if system_type == "trivial":
                    res = trivial_baseline.run(message)
                elif system_type == "keyword":
                    res = keyword_baseline.run(message)
                else:
                    agent = get_pipeline()
                    res = agent.run(message)
                
                res["elapsed_total"] = round(time.time() - t0, 3)
                self.send_json_response(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({"error": str(e)}, status=500)
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def get_taxonomy(self):
        tax_path = PROJECT_ROOT / "src" / "taxonomy" / "intents.json"
        if tax_path.exists():
            with open(tax_path) as f:
                return json.load(f)
        return {"intents": [], "escalation_policy": {}}

    def get_golden_set(self):
        gold_path = PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"
        items = []
        if gold_path.exists():
            with open(gold_path) as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        return items

    def get_stats(self):
        return {
            "brand": "@AppleSupport",
            "pairs_count": 74613,
            "vocab_size": 15000,
            "intents_count": 9,
            "golden_set_size": 200,
            "model": os.getenv("AGENT_MODEL", "qwen/qwen3.8-27b"),
            "provider": os.getenv("DEFAULT_LLM_PROVIDER", "groq"),
            "routing_type": "Deterministic Guardrail Policy",
            "retriever_type": "TF-IDF + Sublinear TF Cosine Sim (74k pairs)",
        }

    def get_eval_report(self):
        report_path = PROJECT_ROOT / "reports" / "eval_report.md"
        results_path = PROJECT_ROOT / "reports" / "eval_results.json"
        raw_json = {}
        report_md = ""
        if results_path.exists():
            with open(results_path) as f:
                raw_json = json.load(f)
        if report_path.exists():
            with open(report_path) as f:
                report_md = f.read()
        return {"metrics": raw_json, "markdown": report_md}


def run_server(port=8080):
    server_address = ("", port)
    # Warm up pipeline on startup
    print(f"🚀 Starting Apple Support AI Agent Dashboard Server on http://localhost:{port}")
    get_pipeline()
    httpd = HTTPServer(server_address, AgentDashboardHandler)
    print(f"✅ Server running and ready at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    run_server(port)
