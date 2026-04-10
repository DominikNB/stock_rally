"""
Fachlogische Stichwörter für GKG-V2Themes (Substring-Vergleich, Großschreibung egal).

Dient nur dem Explorations-Skript: welche der empirisch häufigen Tokens passen zur Sektor-Hypothese.
Keine Garantie, dass GDELT exakt diese Strings nutzt — immer mit BQ-Top-N abgleichen.

GDELT-Doku / Datenmodell: https://www.gdeltproject.org/data.html#documentation
"""
from __future__ import annotations

# Oberbegriffe / Präfixe, die oft im Makro-Kanal vorkommen (Orientierung)
MACRO_THEME_HINTS: list[str] = [
    "ECON_",
    "MONETARY",
    "INFLATION",
    "INTEREST",
    "CENTRALBANK",
    "FED",
    "GDP",
    "TRADE",
    "TAX_",
    "SANCTION",
    "GEOPOL",
]

# Pro Sektor: erwartete Theme-Familien (Erweiterung nach deinen BQ-Ergebnissen)
SECTOR_THEME_HINTS: dict[str, list[str]] = {
    "heat_pump": ["ENV_", "ENERGY", "SUBSID", "GREEN", "HEAT", "HVAC", "CLIMATE"],
    "tech": ["TECH", "AI_", "SEMICON", "SOFTWARE", "CLOUD", "CYBER", "CHIP"],
    "finance": ["BANK", "FINANCE", "ECON_", "MONETARY", "CREDIT", "INSUR", "FED"],
    "healthcare": ["HEALTH", "MEDICAL", "FDA", "PHARM", "CLINICAL", "BIOTECH", "DRUG", "VACCIN"],
    "consumer": ["CONSUMER", "RETAIL", "BRAND", "INFLATION", "WAGE", "LABOR"],
    "industrial": ["MANUFACT", "INDUSTRY", "INDUSTRIAL", "CONSTRUCTION", "MACHINE", "ORDER", "CAPEX"],
    "energy": ["ENERGY", "OIL", "GAS", "PETROL", "OPEC", "POWER", "GRID"],
    "crypto": ["CRYPTO", "BITCOIN", "BLOCKCHAIN", "ETHEREUM", "DEFI"],
    "automotive": ["AUTO", "VEHICLE", "ELECTRIC", "AUTOMOTIVE", "CAR", "EV_"],
    "materials": ["STEEL", "COMMOD", "CHEMIC", "MINING", "METAL"],
    "real_estate": ["REALESTATE", "HOUSING", "MORTGAGE", "PROPERTY", "REIT"],
    "telecom": ["TELECOM", "5G", "BROADBAND", "NETWORK"],
    "media": ["MEDIA", "STREAM", "ADVERT", "BROADCAST"],
}


def hint_hits_for_token(token: str, hints: list[str]) -> list[str]:
    u = token.upper()
    return [h for h in hints if h.upper() in u]
