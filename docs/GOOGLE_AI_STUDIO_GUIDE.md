# Google AI Studio - CelebA Face Intelligence Architecture

This document describes how the CelebA Face Intelligence Suite integrates with **Google AI Studio** for multimodal computer vision and biometric intelligence.

---

## 1. Overview & Architecture

Google AI Studio provides access to Google's flagship Gemini models (`gemini-1.5-flash` and `gemini-1.5-pro`) for multimodal reasoning over face images, facial attributes, expressions, and celebrity metadata.

```
                      +------------------------------------+
                      |    CelebA Face Image (RGB Tensor)  |
                      +------------------------------------+
                                         |
             +---------------------------+---------------------------+
             |                                                       |
             v                                                       v
+-------------------------------+                       +-------------------------------+
|     Google AI Studio API      |                       |    Local PyTorch Checkpoints  |
|   (Gemini 1.5 Multimodal)     |                       |  (Residual CNN, ViT, BiLSTM)  |
|                               |                       |                               |
| • Expression & Smile Rationale|                       | • Quantitative Probability    |
| • Age & Demographics Est.     |                       | • 5-View Test-Time Aug (TTA)  |
| • Facial Hair & Eyeglasses    |                       | • Prediction Uncertainty      |
| • Structured JSON Output      |                       | • Identity-Disjoint Holdout   |
+-------------------------------+                       +-------------------------------+
             |                                                       |
             +---------------------------+---------------------------+
                                         v
                      +------------------------------------+
                      |   Combined Intelligence & Evidence |
                      |    (Auditable & Cross-Validated)   |
                      +------------------------------------+
```

---

## 2. Configuration & Authentication

### Setting up Google AI Studio API Key:
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Set the environment variable in your terminal or `.env` file:
   ```bash
   export GEMINI_API_KEY="your-google-ai-studio-api-key"
   # On Windows PowerShell:
   $env:GEMINI_API_KEY="your-google-ai-studio-api-key"
   ```

### Configuration File (`config/ai_studio_config.yaml`):
```yaml
ai_studio:
  provider: "Google AI Studio"
  default_model: "gemini-1.5-flash"
  alternate_model: "gemini-1.5-pro"
  environment_variable: "GEMINI_API_KEY"

generation_parameters:
  temperature: 0.2
  top_p: 0.95
  max_output_tokens: 1024
  response_mime_type: "application/json"
```

---

## 3. Running the Application

### Command Line Interface:
```bash
python app/google_ai_studio_app.py --image img_align_celeba/000001.png --model gemini-1.5-flash
```

### Python SDK Integration:
```python
from src.ai_studio import GeminiFaceIntelligenceStudio

studio = GeminiFaceIntelligenceStudio(model_name="gemini-1.5-flash")
result = studio.analyze_face("img_align_celeba/000001.png")

print(result["analysis"]["expression"]["is_smiling"])
print(result["analysis"]["demographics"]["apparent_age_group"])
```

---

## 4. Prompt Engineering & System Instructions

The application uses Google AI Studio system instructions configured for biometric analysis:
- **Zero Hallucination Guardrails**: Adheres strictly to the 40 CelebA binary attribute definitions.
- **Multimodal Facial Feature Reasoning**: Cross-references visible facial landmarks with database identities.
- **Structured JSON Mode**: Uses `response_mime_type: "application/json"` to ensure machine-parsable responses.
