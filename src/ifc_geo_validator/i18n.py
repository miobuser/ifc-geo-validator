"""Internationalization (i18n) for Swiss trilingual support.

Provides translations for DE (Deutsch), FR (Français), IT (Italiano).
Default: German (Deutsch).

Usage:
    from ifc_geo_validator.i18n import t, set_language
    set_language("fr")
    print(t("crown_width"))  # → "Largeur de couronnement"
"""

_LANG = "de"

TRANSLATIONS = {
    # ── App UI ────────────────────────────────────────────
    "app_title": {
        "de": "IFC Geometry Validator",
        "fr": "IFC Geometry Validator",
        "it": "IFC Geometry Validator",
    },
    "validation": {
        "de": "Validierung",
        "fr": "Validation",
        "it": "Validazione",
    },
    "ruleset_editor": {
        "de": "Ruleset Editor",
        "fr": "Éditeur de règles",
        "it": "Editor regole",
    },
    "variable_reference": {
        "de": "Variablen-Referenz",
        "fr": "Référence des variables",
        "it": "Riferimento variabili",
    },
    "language": {
        "de": "Sprache",
        "fr": "Langue",
        "it": "Lingua",
    },
    "upload_ifc": {
        "de": "IFC-Datei hochladen",
        "fr": "Télécharger le fichier IFC",
        "it": "Caricare il file IFC",
    },
    "validate": {
        "de": "Validieren",
        "fr": "Valider",
        "it": "Validare",
    },
    "entity_types": {
        "de": "Elementtypen",
        "fr": "Types d'éléments",
        "it": "Tipi di elementi",
    },
    "ruleset": {
        "de": "Regelwerk",
        "fr": "Jeu de règles",
        "it": "Set di regole",
    },
    "elements": {
        "de": "Elemente",
        "fr": "Éléments",
        "it": "Elementi",
    },
    "terrain": {
        "de": "Gelände",
        "fr": "Terrain",
        "it": "Terreno",
    },
    "detected": {
        "de": "Erkannt",
        "fr": "Détecté",
        "it": "Rilevato",
    },
    "none": {
        "de": "Keins",
        "fr": "Aucun",
        "it": "Nessuno",
    },

    # ── Measurements ──────────────────────────────────────
    "volume": {
        "de": "Volumen",
        "fr": "Volume",
        "it": "Volume",
    },
    "surface_area": {
        "de": "Oberfläche",
        "fr": "Surface",
        "it": "Superficie",
    },
    "crown_width": {
        "de": "Kronenbreite",
        "fr": "Largeur de couronnement",
        "it": "Larghezza della corona",
    },
    "crown_slope": {
        "de": "Kronenneigung",
        "fr": "Pente du couronnement",
        "it": "Pendenza della corona",
    },
    "wall_thickness": {
        "de": "Wandstärke",
        "fr": "Épaisseur du mur",
        "it": "Spessore del muro",
    },
    "wall_height": {
        "de": "Wandhöhe",
        "fr": "Hauteur du mur",
        "it": "Altezza del muro",
    },
    "inclination": {
        "de": "Neigung",
        "fr": "Inclinaison",
        "it": "Inclinazione",
    },
    "cross_slope": {
        "de": "Quergefälle",
        "fr": "Dévers",
        "it": "Pendenza trasversale",
    },
    "long_slope": {
        "de": "Längsgefälle",
        "fr": "Pente longitudinale",
        "it": "Pendenza longitudinale",
    },
    "curvature_radius": {
        "de": "Krümmungsradius",
        "fr": "Rayon de courbure",
        "it": "Raggio di curvatura",
    },
    "foundation_width": {
        "de": "Fundamentbreite",
        "fr": "Largeur de fondation",
        "it": "Larghezza della fondazione",
    },
    "embedment_depth": {
        "de": "Einbindetiefe",
        "fr": "Profondeur d'encastrement",
        "it": "Profondità di incorporamento",
    },
    "measurement_uncertainty": {
        "de": "Messunsicherheit",
        "fr": "Incertitude de mesure",
        "it": "Incertezza di misura",
    },
    "watertight": {
        "de": "Wasserdicht",
        "fr": "Étanche",
        "it": "Stagno",
    },
    "curved": {
        "de": "Gekrümmt",
        "fr": "Courbé",
        "it": "Curvo",
    },

    # ── Face categories ───────────────────────────────────
    "crown": {
        "de": "Krone",
        "fr": "Couronnement",
        "it": "Corona",
    },
    "foundation": {
        "de": "Fundament",
        "fr": "Fondation",
        "it": "Fondazione",
    },
    "front": {
        "de": "Ansichtsfläche",
        "fr": "Face vue",
        "it": "Faccia vista",
    },
    "back": {
        "de": "Rückseite",
        "fr": "Face arrière",
        "it": "Faccia posteriore",
    },
    "end_left": {
        "de": "Stirnfläche L",
        "fr": "Face latérale G",
        "it": "Faccia laterale S",
    },
    "end_right": {
        "de": "Stirnfläche R",
        "fr": "Face latérale D",
        "it": "Faccia laterale D",
    },

    # ── Results ───────────────────────────────────────────
    "passed": {
        "de": "Bestanden",
        "fr": "Réussi",
        "it": "Superato",
    },
    "failed": {
        "de": "Fehlgeschlagen",
        "fr": "Échoué",
        "it": "Fallito",
    },
    "skipped": {
        "de": "Übersprungen",
        "fr": "Ignoré",
        "it": "Saltato",
    },
    "check_report": {
        "de": "Prüfprotokoll",
        "fr": "Rapport de contrôle",
        "it": "Rapporto di verifica",
    },
    "project_name": {
        "de": "Projektname",
        "fr": "Nom du projet",
        "it": "Nome del progetto",
    },
    "inspector": {
        "de": "Prüfer/in",
        "fr": "Inspecteur/trice",
        "it": "Ispettore/trice",
    },
    "download_report": {
        "de": "Prüfprotokoll herunterladen",
        "fr": "Télécharger le rapport",
        "it": "Scarica il rapporto",
    },

    # ── 3D Viewer ─────────────────────────────────────────
    "classification_view": {
        "de": "Klassifikation",
        "fr": "Classification",
        "it": "Classificazione",
    },
    "slope_view": {
        "de": "Quergefälle",
        "fr": "Dévers",
        "it": "Pendenza trasversale",
    },
    "ifc_model_view": {
        "de": "IFC-Modell",
        "fr": "Modèle IFC",
        "it": "Modello IFC",
    },

    # ── Element roles ─────────────────────────────────────
    "wall_stem": {
        "de": "Mauerstiel",
        "fr": "Voile de mur",
        "it": "Fusto del muro",
    },
    "parapet": {
        "de": "Brüstung",
        "fr": "Parapet",
        "it": "Parapetto",
    },
    "column": {
        "de": "Stütze",
        "fr": "Colonne",
        "it": "Colonna",
    },
    "slab": {
        "de": "Platte",
        "fr": "Dalle",
        "it": "Soletta",
    },
}


def set_language(lang: str) -> None:
    """Set the active language (de, fr, it)."""
    global _LANG
    if lang in ("de", "fr", "it"):
        _LANG = lang


def get_language() -> str:
    """Get the active language code."""
    return _LANG


def t(key: str) -> str:
    """Translate a key to the active language."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(_LANG, entry.get("de", key))
