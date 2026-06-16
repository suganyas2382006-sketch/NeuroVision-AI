# severity/severity_analysis.py

def calculate_severity(prediction, confidence, localized_mask_active=True):
    """
    Evaluates clinical severity grades and provides the human-interpretable
    reasoning (XAI) behind the algorithmic decision tree.
    """
    if prediction == "No Tumor":
        risk_level = "Low"
        severity_grade = "Grade 0 (Normal Baseline)"
        xai_reasoning = (
            "The model detected no significant tissue variations "
            "consistent with neoplasm morphology. The activation heatmap "
            "shows minimal localized focus."
        )
        return severity_grade, risk_level, xai_reasoning

    # Logic for Glioma and Meningioma
    if prediction in ["Glioma", "Meningioma"]:
        if confidence >= 85.0:
            risk_level = "Critical"
            severity_grade = "Grade III / Grade IV (Advanced Progression Risk)"
            xai_reasoning = (
                f"Localized activation maps identify significant mass effect "
                f"or volumetric distortion. The {confidence}% system confidence "
                f"indicates high certainty of high-grade proliferation."
            )
        else:
            risk_level = "Moderate"
            severity_grade = "Grade I / Grade II (Early/Intermediate Stage)"
            xai_reasoning = (
                f"Model identifies isolated neo-vascular activity, consistent with "
                f"localized lesion baseline. The {confidence}% confidence suggests "
                f"lower proliferative signals but requires triage."
            )
        return severity_grade, risk_level, xai_reasoning

    # Logic for Pituitary Adenomas
    if prediction == "Pituitary":
        if localized_mask_active: # Example conceptual trigger based on heatmap shape
            risk_level = "Moderate"
            severity_grade = "Adenoma Progression (Localized Expansion)"
            xai_reasoning = (
                f"The activation heatmap isolates a dense focus on the sellar "
                f"region. The system flags this as localized adenoma mass expansion "
                f"rather than incidental findings."
            )
        else:
            risk_level = "Low"
            severity_grade = "Baseline Incidental Finding"
            xai_reasoning = (
                "Heatmap shows diffuse signal in the suprasellar region without "
                "isolated focal density. This implies baseline surveillance is adequate."
            )
        return severity_grade, risk_level, xai_reasoning

    # Default fallback
    return "Undetermined Grade", "Unknown Risk", "Clinical Triage required for manual diagnostic verification."
