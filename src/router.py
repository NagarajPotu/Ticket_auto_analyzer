LABEL_TO_TEAM = {
    "Bug Report": "Engineering",
    "Feature Request": "Product",
    "Technical Issue": "Tech Support",
    "Billing Inquiry": "Finance",
    "Account Management": "Customer Success"
}

def route_ticket(pred_label: str, score: float, priority: str = None, vip: bool = False):
    if vip:
        return "VIP Queue"
    
    # Always return predicted team, add review note for mid-score if desired
    if score >= 0.85:
        return LABEL_TO_TEAM.get(pred_label, "Human Triage Queue")
    elif score >= 0.4:   # Lowered threshold to catch more tickets
        return f"{LABEL_TO_TEAM.get(pred_label, 'Human Triage Queue')} (Review Required)"
    
    # For very low confidence
    return "Human Triage Queue"
