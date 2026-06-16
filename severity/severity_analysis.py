def calculate_severity(prediction, confidence):
    """
    Evaluates clinical severity grade matrices based on predicted diagnosis types
    and algorithmic model processing thresholds.
    """
    if prediction == "No Tumor":
        return "Normal / No Anomalies Detected", "Low"
        
    if prediction in ["Glioma", "Meningioma"]:
        if confidence >= 85.0:
            return "Grade III / Grade IV (High Advanced Progression Risk)", "Critical"
        else:
            return "Grade I / Grade II (Early Stage Progression)", "Moderate"
            
    if prediction == "Pituitary":
        if confidence >= 80.0:
            return "Adenoma Progression (Requires Localized Evaluation)", "Moderate"
        else:
            return "Microadenoma / Incidentaloma Baseline", "Low"
            
    return "Undetermined Course - Awaiting Manual Triage", "Unknown"
