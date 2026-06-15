def get_severity(confidence):
    if confidence < 40:
        return "Low"
    elif confidence < 70:
        return "Medium"
    else:
        return "High"