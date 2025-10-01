# %%
import pandas as pd
import numpy as np
import time
import json
from typing import Iterable, Dict, Any, List, Optional
import requests
import seaborn as sns
import networkx as nx


# %%
import requests

def _props_dict(item):
    props = {}
    plist = (item.get("propertyConceptList") or {}).get("propertyConcept") or []
    for p in plist:
        k = (p.get("propName") or "").strip()
        v = (p.get("propValue") or "").strip()
        if k:
            props[k.upper()] = v
    return props

def _first_match(props, *keys_or_contains):
    for key in keys_or_contains:
        if key in props:
            return props[key]
    for key in props:
        for needle in keys_or_contains:
            if needle in key:
                return props[key]
    return None

def _rxnorm_names_from_rxcui(rxcui):
    base = "https://rxnav.nlm.nih.gov/REST"
    url = f"{base}/rxcui/{rxcui}/allProperties.json"
    resp = requests.get(url, params={"prop": "names"}, timeout=10)
    resp.raise_for_status()
    data = resp.json() or {}
    props = (data.get("propConceptGroup") or {}).get("propConcept") or []
    names = [(p.get("propName"), p.get("propValue")) for p in props]

    brand = None
    for code, val in names:
        if (code or "").upper() in ("BN", "SBD", "SBDF", "SBDG") and val:
            brand = val
            break

    generic = None
    for code, val in names:
        if (code or "").upper() in ("IN", "SCD", "SCDF", "SCDC") and val:
            generic = val
            break

    if generic is None:
        url2 = f"{base}/rxcui/{rxcui}/property.json"
        r2 = requests.get(url2, params={"propName": "RxNorm Name"}, timeout=10)
        if r2.ok:
            dd = r2.json() or {}
            generic = (dd.get("propConceptGroup") or {}).get("propConcept", [{}])[0].get("propValue")

    return brand, generic

def get_names_for_ndc(ndc_code, ndcstatus="ALL"):
    """
    Returns a dict with:
      - ndc11
      - brand_name
      - product_type
      - generic_name
      - labeler
      - rxcui
    Uses RxCUI fallbacks to fill brand/generic when missing in NDC properties.
    """
    ndc_url = "https://rxnav.nlm.nih.gov/REST/ndcproperties.json"
    r = requests.get(ndc_url, params={"id": ndc_code, "ndcstatus": ndcstatus}, timeout=10)
    r.raise_for_status()
    data = r.json() or {}
    items = (data.get("ndcPropertyList") or {}).get("ndcProperty") or []
    if not items:
        return {
            "ndc11": None,
            "brand_name": None,
            "product_type": None,
            "generic_name": None,
            "labeler": None,
            "rxcui": None,
        }

    # pick richest item (prefer one with explicit proprietary/nonproprietary if present)
    best = None
    for it in items:
        props = _props_dict(it)
        if "PROPRIETARYNAME" in props or "NONPROPRIETARYNAME" in props:
            best = it
            break
    if best is None:
        best = items[0]

    props = _props_dict(best)
    rxcui = best.get("rxcui")
    ndc11 = best.get("ndcItem")  # RxNav’s NDC11 field

    # direct reads
    brand = _first_match(props, "PROPRIETARYNAME", "PROPRIETARY NAME", "PROPRIETARY")
    generic = _first_match(props, "NONPROPRIETARYNAME", "NONPROPRIETARY NAME", "NONPROPRIETARY")
    product_type = _first_match(props, "PRODUCTTYPENAME", "PRODUCT TYPE")
    labeler = _first_match(props, "LABELER", "LABELERNAME", "LABELER NAME")

    # fallbacks via RxCUI
    if (not brand or not generic) and rxcui:
        rx_brand, rx_generic = _rxnorm_names_from_rxcui(rxcui)
        brand = brand or rx_brand
        generic = generic or rx_generic

    return {
        "ndc11": ndc11,
        "brand_name": brand,
        "product_type": product_type,   # e.g., HUMAN PRESCRIPTION DRUG
        "generic_name": generic,
        "labeler": labeler,
        "rxcui": rxcui,
    }

# ---- Example usage
if __name__ == "__main__":
    summary = get_names_for_ndc("00003-0894", ndcstatus="ALL")

print("NDC11:", summary["ndc11"])
print("Brand name:", summary["brand_name"])
print("Product type:", summary["product_type"])
print("Generic name:", summary["generic_name"])
print("Labeler:", summary["labeler"])
print("RxCUI:", summary["rxcui"])


# %%



