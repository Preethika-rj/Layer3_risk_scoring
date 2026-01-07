"""
FastAPI CVE Analysis Service
Refactored from standalone script to production-ready API endpoint
"""

import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64

# ===================== CONFIGURATION =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store API configuration - should be moved to environment variables in production
class Config:
    OPENROUTER_API_KEY: str = "YOUR_API_KEY_HERE"  # Use env var: os.getenv("OPENROUTER_API_KEY")
    MODEL_NAME: str = "openai/gpt-4o-mini"
    OPENROUTER_ENDPOINT: str = "https://openrouter.ai/api/v1/chat/completions"
    REQUEST_TIMEOUT: int = 30

config = Config()

# ===================== PYDANTIC MODELS =====================

class CVEInput(BaseModel):
    """Request model for CVE analysis"""
    cve_id: str = Field(..., description="CVE identifier", example="CVE-2026-21428")
    cvss_vector: Optional[str] = Field(None, description="CVSS vector string")
    severity: str = Field(..., description="Severity level", example="HIGH")
    attack_vector: Optional[str] = Field(None, description="Attack vector type")
    cvss_score: float = Field(..., ge=0.0, le=10.0, description="CVSS score between 0 and 10")
    summary: str = Field(..., description="Brief vulnerability summary")
    description: str = Field(..., description="Detailed vulnerability description")
    affected_assets: List[str] = Field(default_factory=list, description="List of affected assets")

    @validator('cvss_score')
    def validate_cvss_score(cls, v):
        if not 0.0 <= v <= 10.0:
            raise ValueError('CVSS score must be between 0.0 and 10.0')
        return round(v, 1)

    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NONE']
        if v.upper() not in valid_severities:
            raise ValueError(f'Severity must be one of: {", ".join(valid_severities)}')
        return v.upper()


class EPSSData(BaseModel):
    """EPSS prediction data"""
    score: float = Field(..., description="Current EPSS score")
    predicted_30d: float = Field(..., description="Predicted EPSS score in 30 days")


class CVEAnalysisResponse(BaseModel):
    """Response model for CVE analysis"""
    cve_id: str
    cvss_vector: Optional[str]
    severity: str
    cvss_score: float
    simple_summary: str
    simple_description: str
    affected_products: List[str]
    affected_assets: List[str]
    fixes: List[str]
    epss_score: float
    epss_30d_prediction: float
    cvss_heatmap_base64: Optional[str] = Field(None, description="Base64 encoded CVSS heatmap image")
    epss_plot_base64: Optional[str] = Field(None, description="Base64 encoded EPSS prediction plot")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


# ===================== CORE LOGIC CLASSES =====================

class ProductDetector:
    """Detects affected products from CVE data"""
    
    KNOWN_PRODUCTS = [
        "openssh", "red hat enterprise linux", "rhel", "linux",
        "apache", "nginx", "windows", "mysql", "postgresql"
    ]
    
    @classmethod
    def detect_products(cls, cve_data: Dict[str, Any]) -> List[str]:
        """
        Extract products mentioned in CVE descriptions
        
        Args:
            cve_data: Dictionary containing CVE information
            
        Returns:
            List of detected product names
        """
        text = (
            cve_data.get("simple_description", "") +
            cve_data.get("simple_summary", "") +
            " ".join(cve_data.get("affected_assets", []))
        ).lower()

        found = []
        for product in cls.KNOWN_PRODUCTS:
            if product in text:
                found.append(product.upper())

        # Return unique products or default fallback
        return list(set(found)) if found else ["OpenSSH", "Red Hat Enterprise Linux"]


class FixGenerator:
    """Generates remediation fixes for detected products"""
    
    @staticmethod
    def generate_fixes(products: List[str]) -> List[str]:
        """
        Generate fix recommendations based on affected products
        
        Args:
            products: List of affected product names
            
        Returns:
            List of remediation steps
        """
        fixes = []
        
        # Product-specific fixes
        for product in products:
            fixes.append(f"Update or replace {product} with the latest secure version from the vendor.")

        # General security recommendations
        fixes.extend([
            "Install updates only from official vendor sources.",
            "Remove any software obtained from unknown or untrusted websites.",
            "Ask your system administrator to confirm the system is safe."
        ])
        
        return fixes


class CVESummarizer:
    """LLM-based CVE summarizer using OpenRouter API"""
    
    def __init__(self, api_key: str, model: str, endpoint: str, timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def summarize(self, cve: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate simplified summary using LLM
        
        Args:
            cve: CVE data dictionary
            
        Returns:
            Dictionary with simple_summary and simple_description
            
        Raises:
            HTTPException: If API call fails
        """
        prompt = self._build_prompt(cve)
        
        try:
            response = self._call_api(prompt)
            data = self._parse_response(response)
            return data
        except requests.exceptions.Timeout:
            logger.error("LLM API timeout")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM summarization service timed out"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM summarization service unavailable"
            )
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return fallback data instead of failing
            return {
                "simple_summary": cve.get("summary", ""),
                "simple_description": cve.get("description", "")
            }

    def _build_prompt(self, cve: Dict[str, Any]) -> str:
        """Build the LLM prompt"""
        return (
            "Rewrite this vulnerability in SIMPLE English.\n"
            "Return ONLY JSON:\n"
            "{"
            '"simple_summary":"","simple_description":""'
            "}\n\n"
            f"{json.dumps(cve)}"
        )

    def _call_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to LLM service"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM API response"""
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)


class EPSSPredictor:
    """EPSS (Exploit Prediction Scoring System) calculator"""
    
    @staticmethod
    def predict_epss(cvss_vector: Optional[str] = None) -> EPSSData:
        """
        Calculate EPSS score and 30-day prediction
        
        Args:
            cvss_vector: CVSS vector string
            
        Returns:
            EPSSData with current and predicted scores
        """
        score = 0.0008
        
        if cvss_vector:
            v = cvss_vector.upper()
            if "AV:N" in v:  # Network attack vector
                score *= 1.6
            if "AC:L" in v:  # Low attack complexity
                score *= 1.4
        
        score = round(score, 5)
        predicted = round(score * 1.18, 5)
        
        return EPSSData(score=score, predicted_30d=predicted)


class VisualizationGenerator:
    """Generate visualizations for CVE analysis"""
    
    @staticmethod
    def generate_cvss_heatmap(score: float, cve_id: str) -> str:
        """
        Generate CVSS heatmap and return as base64 string
        
        Args:
            score: CVSS score
            cve_id: CVE identifier
            
        Returns:
            Base64 encoded PNG image
        """
        colors = [(0, "green"), (0.4, "yellow"), (0.7, "orange"), (1, "red")]
        cmap = mcolors.LinearSegmentedColormap.from_list("cvss", colors)
        norm = mcolors.Normalize(vmin=0, vmax=10)

        fig, ax = plt.subplots(figsize=(5, 1.5))
        im = ax.imshow([[score]], cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"CVSS Heatmap — {cve_id}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0, 0, str(score), ha="center", va="center", 
                fontsize=14, fontweight="bold")

        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.4, pad=0.3)
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_base64

    @staticmethod
    def generate_epss_plot(current: float, predicted: float, cve_id: str) -> str:
        """
        Generate EPSS prediction plot and return as base64 string
        
        Args:
            current: Current EPSS score
            predicted: Predicted EPSS score
            cve_id: CVE identifier
            
        Returns:
            Base64 encoded PNG image
        """
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
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_base64


# ===================== FASTAPI APPLICATION =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info("Starting CVE Analysis API...")
    if config.OPENROUTER_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("⚠️  OpenRouter API key not configured! Set OPENROUTER_API_KEY environment variable.")
    yield
    # Shutdown
    logger.info("Shutting down CVE Analysis API...")


app = FastAPI(
    title="CVE Analysis API",
    description="Analyze CVE vulnerabilities with AI-powered summaries, EPSS predictions, and visualizations",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "CVE Analysis API",
        "status": "operational",
        "version": "1.0.0"
    }


@app.post(
    "/analyze",
    response_model=CVEAnalysisResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        503: {"model": ErrorResponse, "description": "External service unavailable"},
        504: {"model": ErrorResponse, "description": "Request timeout"}
    }
)
async def analyze_cve(cve_input: CVEInput, include_visualizations: bool = True):
    """
    Analyze a CVE vulnerability
    
    This endpoint performs comprehensive CVE analysis including:
    - AI-powered simplification of technical descriptions
    - Product detection and remediation suggestions
    - EPSS exploit prediction scoring
    - Optional visualization generation
    
    Args:
        cve_input: CVE data to analyze
        include_visualizations: Whether to generate and return base64 encoded plots
        
    Returns:
        CVEAnalysisResponse with complete analysis results
    """
    try:
        logger.info(f"Analyzing CVE: {cve_input.cve_id}")
        
        # Step 1: Generate AI summary
        summarizer = CVESummarizer(
            api_key=config.OPENROUTER_API_KEY,
            model=config.MODEL_NAME,
            endpoint=config.OPENROUTER_ENDPOINT,
            timeout=config.REQUEST_TIMEOUT
        )
        
        # Convert input to dict for summarizer
        cve_dict = cve_input.dict()
        ai_summary = summarizer.summarize(cve_dict)
        
        # Step 2: Predict EPSS
        epss_data = EPSSPredictor.predict_epss(cve_input.cvss_vector)
        
        # Step 3: Detect products and generate fixes
        # Merge AI summary with original data for product detection
        detection_data = {**cve_dict, **ai_summary}
        affected_products = ProductDetector.detect_products(detection_data)
        fixes = FixGenerator.generate_fixes(affected_products)
        
        # Step 4: Generate visualizations if requested
        cvss_heatmap = None
        epss_plot = None
        
        if include_visualizations:
            try:
                cvss_heatmap = VisualizationGenerator.generate_cvss_heatmap(
                    cve_input.cvss_score,
                    cve_input.cve_id
                )
                epss_plot = VisualizationGenerator.generate_epss_plot(
                    epss_data.score,
                    epss_data.predicted_30d,
                    cve_input.cve_id
                )
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
                # Continue without visualizations rather than failing
        
        # Step 5: Build response
        response = CVEAnalysisResponse(
            cve_id=cve_input.cve_id,
            cvss_vector=cve_input.cvss_vector,
            severity=cve_input.severity,
            cvss_score=cve_input.cvss_score,
            simple_summary=ai_summary.get("simple_summary", cve_input.summary),
            simple_description=ai_summary.get("simple_description", cve_input.description),
            affected_products=affected_products,
            affected_assets=cve_input.affected_assets,
            fixes=fixes,
            epss_score=epss_data.score,
            epss_30d_prediction=epss_data.predicted_30d,
            cvss_heatmap_base64=cvss_heatmap,
            epss_plot_base64=epss_plot
        )
        
        logger.info(f"Successfully analyzed CVE: {cve_input.cve_id}")
        return response
        
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error analyzing CVE: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ===================== ADDITIONAL ENDPOINTS =====================

@app.post("/epss/predict", response_model=EPSSData)
async def predict_epss(cvss_vector: Optional[str] = None):
    """
    Predict EPSS score based on CVSS vector
    
    Args:
        cvss_vector: Optional CVSS vector string
        
    Returns:
        EPSS score and 30-day prediction
    """
    return EPSSPredictor.predict_epss(cvss_vector)



# ===================== EXCEPTION HANDLERS =====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)