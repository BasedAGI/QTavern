def format_quant_type(qtype: str) -> str:
    """
    Formats the quantization method string for the repository name.
    """
    if qtype.lower().startswith("i1-"):
        parts = qtype.split("-", 1)
        if len(parts) == 2:
            return "i1-" + parts[1].upper()
        else:
            return qtype.lower()
    else:
        return qtype.upper()