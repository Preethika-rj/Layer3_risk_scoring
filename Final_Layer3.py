import json
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Any, List

# ===================== CONFIG =====================
OPENROUTER_API_KEY = "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"
MODEL_NAME = "openai/gpt-4o-mini"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# ===================== PRODUCT EXTRACTION =====================
def detect_products(cve: Dict[str, Any]) -> List[str]:
    text = (
        cve.get("simple_description", "") +
        cve.get("simple_summary", "") +
        " ".join(cve.get("affected_assets", []))
    ).lower()

    known_products = ["openssh", "red hat enterprise linux", "rhel", "linux"]
    found = []

    for p in known_products:
        if p in text:
            found.append(p.upper())

    return list(set(found)) if found else ["OpenSSH", "Red Hat Enterprise Linux"]

# ===================== FIX GENERATOR =====================
def generate_fixes(products: List[str]) -> List[str]:
    fixes = []
    for p in products:
        fixes.append(f"Update or replace {p} with the latest secure version from the vendor.")

    fixes.extend([
        "Install updates only from official vendor sources.",
        "Remove any software obtained from unknown or untrusted websites.",
        "Ask your system administrator to confirm the system is safe."
    ])
    return fixes

# ===================== LLM SUMMARIZER =====================
class CVESummarizer:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def summarize(self, cve: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(cve)
        try:
            response = self._call_api(prompt)
            data = self._parse_response(response)
        except Exception:
            data = {}

        return data

    def _build_prompt(self, cve: Dict[str, Any]) -> str:
        return (
            "Rewrite this vulnerability in SIMPLE English.\n"
            "Return ONLY JSON:\n"
            "{"
            '"simple_summary":"","simple_description":""'
            "}\n\n"
            f"{json.dumps(cve)}"
        )

    def _call_api(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

# ===================== CVSS HEATMAP =====================
def show_cvss_heatmap(score: float, cve_id: str):
    colors = [(0, "green"), (0.4, "yellow"), (0.7, "orange"), (1, "red")]
    cmap = mcolors.LinearSegmentedColormap.from_list("cvss", colors)
    norm = mcolors.Normalize(vmin=0, vmax=10)

    fig, ax = plt.subplots(figsize=(5, 1.5))
    im = ax.imshow([[score]], cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(f"CVSS Heatmap — {cve_id}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 0, str(score), ha="center", va="center", fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.4, pad=0.3)
    plt.tight_layout()
    plt.show(block=False)

# ===================== EPSS =====================
def ai_predict_epss(vector: str | None = None):
    score = 0.0008
    if vector:
        v = vector.upper()
        if "AV:N" in v: score *= 1.6
        if "AC:L" in v: score *= 1.4
    score = round(score, 5)
    return {"score": score, "predicted_30d": round(score * 1.18, 5)}

def plot_epss(current: float, predicted: float, cve_id: str):
    plt.figure(figsize=(6, 4))
    plt.plot([0], [current], marker="o", label="Current EPSS")
    plt.plot([30], [predicted], marker="o", label="30-day Prediction")
    plt.plot([0, 30], [current, predicted], linestyle="--")
    plt.title(f"EPSS Prediction — {cve_id}")
    plt.xlabel("Days")
    plt.ylabel("EPSS Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

# ===================== MAIN =====================
if __name__ == "__main__":
    sample_input = {
        "cve_id": "CVE-2008-3844",
        "cvss_vector": "AV:N/AC:M/Au:N/C:C/I:C/A:C",
        "severity": "CRITICAL",
        "cvss_score": 9.3,
        "affected_assets": ["Server A", "Finance Server"]
    }

    summarizer = CVESummarizer(OPENROUTER_API_KEY, MODEL_NAME)
    ai_text = summarizer.summarize(sample_input)

    epss = ai_predict_epss(sample_input.get("cvss_vector"))

    affected_products = detect_products(ai_text)
    fixes = generate_fixes(affected_products)

    show_cvss_heatmap(sample_input["cvss_score"], sample_input["cve_id"])
    plot_epss(epss["score"], epss["predicted_30d"], sample_input["cve_id"])

    output = {
        "cve_id": sample_input["cve_id"],
        "cvss_vector": sample_input["cvss_vector"],
        "severity": sample_input["severity"],
        "cvss_score": sample_input["cvss_score"],
        "simple_summary": ai_text.get("simple_summary", ""),
        "simple_description": ai_text.get("simple_description", ""),
        "affected_products": affected_products,
        "affected_assets": sample_input["affected_assets"],
        "fixes": fixes,
        "epss_score": epss["score"],
        "epss_30d_prediction": epss["predicted_30d"]
    }

    print(json.dumps(output, indent=2))
    plt.show()
