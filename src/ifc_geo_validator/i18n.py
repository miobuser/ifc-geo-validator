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
        "it": "Larghezza del coronamento",
    },
    "crown_slope": {
        "de": "Kronenneigung",
        "fr": "Pente du couronnement",
        "it": "Pendenza del coronamento",
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
        "it": "Profondità d'infissione",
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
        "it": "Coronamento",
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
        "fr": "Inspecteur·trice",
        "it": "Ispettore/Ispettrice",
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
        "fr": "Voile",
        "it": "Elevazione",
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

    # ── Sidebar & navigation ─────────────────────────────
    "navigation": {"de": "Navigation", "fr": "Navigation", "it": "Navigazione"},
    "files_not_stored": {
        "de": "Dateien werden nicht gespeichert. Verarbeitung nur im Arbeitsspeicher.",
        "fr": "Les fichiers ne sont pas stockés. Traitement uniquement en mémoire.",
        "it": "I file non vengono memorizzati. Elaborazione solo in memoria.",
    },
    "upload_ifc_help": {
        "de": "IFC 4x3 Modell mit geometrischen Elementen",
        "fr": "Modèle IFC 4x3 avec éléments géométriques",
        "it": "Modello IFC 4x3 con elementi geometrici",
    },
    "entity_types_help": {
        "de": "Einen oder mehrere IFC-Elementtypen für die Prüfung wählen",
        "fr": "Sélectionner un ou plusieurs types d'éléments IFC à valider",
        "it": "Selezionare uno o più tipi di elementi IFC da validare",
    },
    "predefined_type": {
        "de": "Predefined Type (optional)",
        "fr": "Type prédéfini (optionnel)",
        "it": "Tipo predefinito (opzionale)",
    },
    "custom_ruleset_upload": {
        "de": "Eigenes Ruleset (YAML)",
        "fr": "Jeu de règles personnalisé (YAML)",
        "it": "Set di regole personalizzato (YAML)",
    },
    "custom_upload": {"de": "Eigenes (Upload)", "fr": "Personnalisé (upload)", "it": "Personalizzato (upload)"},

    # ── Welcome / intro ──────────────────────────────────
    "intro_text": {
        "de": "Geometrische Prüfung von IFC-Infrastrukturmodellen gegen konfigurierbare Anforderungen (ASTRA FHB T/G — Stützmauern).",
        "fr": "Vérification géométrique des modèles IFC d'infrastructure selon des exigences configurables (ASTRA FHB T/G — murs de soutènement).",
        "it": "Verifica geometrica di modelli IFC infrastrutturali secondo requisiti configurabili (ASTRA FHB T/G — muri di sostegno).",
    },
    "pipeline_title": {"de": "Validierungs-Pipeline", "fr": "Pipeline de validation", "it": "Pipeline di validazione"},
    "level_col": {"de": "Stufe", "fr": "Niveau", "it": "Livello"},
    "description_col": {"de": "Beschreibung", "fr": "Description", "it": "Descrizione"},
    "output_col": {"de": "Ergebnis", "fr": "Résultat", "it": "Risultato"},
    "upload_to_begin": {
        "de": "IFC-Datei in der Seitenleiste hochladen, um zu beginnen.",
        "fr": "Télécharger un fichier IFC dans la barre latérale pour commencer.",
        "it": "Caricare un file IFC nella barra laterale per iniziare.",
    },
    "load_demo_model": {
        "de": "📦 Demo-Modell laden",
        "fr": "📦 Charger un modèle de démo",
        "it": "📦 Carica modello demo",
    },
    "demo_model_caption": {
        "de": "Kein IFC zur Hand? Probiere das ASTRA-konforme T28-Showcase-Modell.",
        "fr": "Pas de fichier IFC ? Essayez le modèle T28 conforme ASTRA.",
        "it": "Nessun file IFC? Prova il modello T28 conforme ASTRA.",
    },
    "overall_verdict": {
        "de": "Gesamtbewertung",
        "fr": "Verdict global",
        "it": "Verdetto complessivo",
    },
    "verdict_pass": {"de": "✅ KONFORM", "fr": "✅ CONFORME", "it": "✅ CONFORME"},
    "verdict_fail": {
        "de": "❌ NICHT KONFORM",
        "fr": "❌ NON CONFORME",
        "it": "❌ NON CONFORME",
    },
    "verdict_partial": {
        "de": "⚠️ TEILWEISE KONFORM",
        "fr": "⚠️ PARTIELLEMENT CONFORME",
        "it": "⚠️ PARZIALMENTE CONFORME",
    },
    "dl_json": {
        "de": "JSON-Report (strukturiert)",
        "fr": "Rapport JSON (structuré)",
        "it": "Rapporto JSON (strutturato)",
    },
    "dl_enriched": {
        "de": "Angereicherte IFC (mit Pset_GeoValidation)",
        "fr": "IFC enrichi (avec Pset_GeoValidation)",
        "it": "IFC arricchito (con Pset_GeoValidation)",
    },
    "dl_bcf": {
        "de": "BCF-Issues (für Revit / Navisworks)",
        "fr": "Problèmes BCF (pour Revit / Navisworks)",
        "it": "Problemi BCF (per Revit / Navisworks)",
    },

    # ── Summary panel ────────────────────────────────────
    "validated": {"de": "Validiert", "fr": "Validé", "it": "Validato"},
    "errors": {"de": "Fehler", "fr": "Erreurs", "it": "Errori"},
    "rules_passed": {"de": "Regeln bestanden", "fr": "Règles réussies", "it": "Regole superate"},
    "element_overview": {"de": "Element-Übersicht", "fr": "Aperçu des éléments", "it": "Panoramica elementi"},
    "select_element_for_detail": {
        "de": "Element für Detailansicht wählen",
        "fr": "Sélectionner un élément pour le détail",
        "it": "Selezionare un elemento per i dettagli",
    },
    "coordinate_system": {"de": "Koordinatensystem", "fr": "Système de coordonnées", "it": "Sistema di coordinate"},
    "crs_not_declared": {
        "de": "keine IfcProjectedCRS im Modell deklariert",
        "fr": "aucune IfcProjectedCRS déclarée dans le modèle",
        "it": "nessuna IfcProjectedCRS dichiarata nel modello",
    },

    # ── Level / section headings ─────────────────────────
    "geometry_l1": {"de": "Geometrie (L1)", "fr": "Géométrie (L1)", "it": "Geometria (L1)"},
    "face_classification_l2": {
        "de": "Flächen-Klassifikation (L2)",
        "fr": "Classification des faces (L2)",
        "it": "Classificazione delle facce (L2)",
    },
    "measurements_l3": {"de": "Messwerte (L3)", "fr": "Mesures (L3)", "it": "Misurazioni (L3)"},
    "rule_checks_l4": {"de": "Regelprüfung (L4)", "fr": "Contrôle des règles (L4)", "it": "Controllo regole (L4)"},
    "inter_element_l5": {
        "de": "Inter-Element (L5)",
        "fr": "Inter-éléments (L5)",
        "it": "Inter-elementi (L5)",
    },
    "terrain_context_l6": {
        "de": "Terrain-Kontext (L6)",
        "fr": "Contexte du terrain (L6)",
        "it": "Contesto del terreno (L6)",
    },
    "anomalies": {"de": "Anomalien", "fr": "Anomalies", "it": "Anomalie"},

    # ── Common metric labels ─────────────────────────────
    "triangles": {"de": "Dreiecke", "fr": "Triangles", "it": "Triangoli"},
    "vertices": {"de": "Vertices", "fr": "Sommets", "it": "Vertici"},
    "bounding_box": {"de": "Hüllquader", "fr": "Boîte englobante", "it": "Riquadro di delimitazione"},
    "yes": {"de": "Ja", "fr": "Oui", "it": "Sì"},
    "no": {"de": "Nein", "fr": "Non", "it": "No"},
    "value_actual": {"de": "Ist", "fr": "Mesuré", "it": "Effettivo"},
    "value_expected": {"de": "Soll", "fr": "Attendu", "it": "Atteso"},
    "role": {"de": "Rolle", "fr": "Rôle", "it": "Ruolo"},
    "status": {"de": "Status", "fr": "Statut", "it": "Stato"},
    "unknown": {"de": "unbekannt", "fr": "inconnu", "it": "sconosciuto"},

    # ── Ruleset editor ───────────────────────────────────
    "ruleset_name": {"de": "Ruleset-Name", "fr": "Nom du jeu de règles", "it": "Nome del set di regole"},
    "available_variables": {"de": "Verfügbare Variablen", "fr": "Variables disponibles", "it": "Variabili disponibili"},
    "new_rule": {"de": "Neue Regel", "fr": "Nouvelle règle", "it": "Nuova regola"},
    "variable": {"de": "Variable", "fr": "Variable", "it": "Variabile"},
    "operator": {"de": "Operator", "fr": "Opérateur", "it": "Operatore"},
    "threshold": {"de": "Schwellwert", "fr": "Seuil", "it": "Soglia"},
    "severity": {"de": "Schweregrad", "fr": "Sévérité", "it": "Severità"},
    "add_rule": {"de": "Regel hinzufügen", "fr": "Ajouter une règle", "it": "Aggiungi regola"},
    "rules_count": {"de": "Regeln", "fr": "Règles", "it": "Regole"},
    "download_yaml": {"de": "YAML herunterladen", "fr": "Télécharger YAML", "it": "Scarica YAML"},
    "all_available_variables": {
        "de": "Alle verfügbaren Variablen für YAML-Regeln",
        "fr": "Toutes les variables disponibles pour les règles YAML",
        "it": "Tutte le variabili disponibili per le regole YAML",
    },

    # ── Viewer toolbar ───────────────────────────────────
    "tb_group_color": {"de": "Farbe", "fr": "Couleur", "it": "Colore"},
    "tb_group_view": {"de": "Ansicht", "fr": "Vue", "it": "Vista"},
    "tb_group_display": {"de": "Anzeige", "fr": "Affichage", "it": "Visualizzazione"},
    "tb_group_tool": {"de": "Werkzeug", "fr": "Outil", "it": "Strumento"},
    "tb_group_section": {"de": "Schnitt", "fr": "Coupe", "it": "Sezione"},
    "tb_group_focus": {"de": "Fokus", "fr": "Focus", "it": "Focus"},
    "tb_status": {"de": "Status", "fr": "Statut", "it": "Stato"},
    "tb_faces": {"de": "Flächen", "fr": "Faces", "it": "Facce"},
    "tb_role": {"de": "Rolle", "fr": "Rôle", "it": "Ruolo"},
    "tb_solid": {"de": "Einfarbig", "fr": "Uni", "it": "Uniforme"},
    "tb_fit": {"de": "Einpassen", "fr": "Ajuster", "it": "Adatta"},
    "tb_iso": {"de": "Iso", "fr": "Iso", "it": "Iso"},
    "tb_top": {"de": "Oben", "fr": "Dessus", "it": "Sopra"},
    "tb_front": {"de": "Vorne", "fr": "Avant", "it": "Avanti"},
    "tb_side": {"de": "Seite", "fr": "Côté", "it": "Lato"},
    "tb_wire": {"de": "Gitter", "fr": "Filaire", "it": "Wireframe"},
    "tb_edges": {"de": "Kanten", "fr": "Arêtes", "it": "Bordi"},
    "tb_terrain": {"de": "Gelände", "fr": "Terrain", "it": "Terreno"},
    "tb_ghost": {"de": "Transparent", "fr": "Fantôme", "it": "Trasparente"},
    "tb_measure": {"de": "Messen", "fr": "Mesurer", "it": "Misura"},
    "tb_section_off": {"de": "Aus", "fr": "Arrêt", "it": "Disattiva"},
    "tb_zoom_selection": {"de": "Zoom", "fr": "Zoom", "it": "Zoom"},
    "tb_clear": {"de": "Löschen", "fr": "Effacer", "it": "Pulisci"},
    "tb_flip": {"de": "Umkehren", "fr": "Inverser", "it": "Inverti"},

    # ── Viewer tooltips ──────────────────────────────────
    "tt_status_mode": {
        "de": "Validierungs-Status",
        "fr": "État de validation",
        "it": "Stato di validazione",
    },
    "tt_faces_mode": {
        "de": "Flächen-Klassifikation (L2)",
        "fr": "Classification des faces (L2)",
        "it": "Classificazione delle facce (L2)",
    },
    "tt_role_mode": {"de": "Element-Rolle (L2)", "fr": "Rôle de l'élément (L2)", "it": "Ruolo dell'elemento (L2)"},
    "tt_solid_mode": {"de": "Einheitliche Farbe", "fr": "Couleur uniforme", "it": "Colore uniforme"},
    "tt_ghost_tip": {
        "de": "Andere ausblenden",
        "fr": "Masquer les autres",
        "it": "Nascondi gli altri",
    },
    "tt_measure_tip": {"de": "Strecke messen", "fr": "Mesurer la distance", "it": "Misura distanza"},
    "tt_sections_clear": {
        "de": "Alle Schnitte aufheben",
        "fr": "Effacer toutes les coupes",
        "it": "Rimuovi tutte le sezioni",
    },
    "tt_flip_dir": {"de": "Richtung umkehren", "fr": "Inverser la direction", "it": "Inverti la direzione"},
    "tt_zoom_selection": {
        "de": "Zu Auswahl zoomen",
        "fr": "Zoomer sur la sélection",
        "it": "Zoom sulla selezione",
    },
    "tt_clear_selection": {
        "de": "Auswahl löschen",
        "fr": "Effacer la sélection",
        "it": "Deseleziona",
    },

    # ── Viewer panels ────────────────────────────────────
    "panel_elements": {"de": "Elemente", "fr": "Éléments", "it": "Elementi"},
    "panel_properties": {"de": "Eigenschaften", "fr": "Propriétés", "it": "Proprietà"},
    "panel_collapse": {"de": "Einklappen", "fr": "Réduire", "it": "Riduci"},
    "panel_expand_elements": {
        "de": "Elemente anzeigen",
        "fr": "Afficher les éléments",
        "it": "Mostra elementi",
    },
    "panel_expand_props": {
        "de": "Eigenschaften anzeigen",
        "fr": "Afficher les propriétés",
        "it": "Mostra proprietà",
    },
    "click_element_to_inspect": {
        "de": "Element anklicken zum Inspizieren",
        "fr": "Cliquer sur un élément pour l'inspecter",
        "it": "Clicca un elemento per ispezionarlo",
    },
    "measurements_section": {"de": "Messwerte", "fr": "Mesures", "it": "Misure"},
    "rule_checks_section": {
        "de": "Regelprüfung",
        "fr": "Contrôle des règles",
        "it": "Controllo regole",
    },
    "no_rules_evaluated": {
        "de": "Keine Regeln evaluiert",
        "fr": "Aucune règle évaluée",
        "it": "Nessuna regola valutata",
    },
    "no_mesh_data": {
        "de": "Keine Mesh-Daten verfügbar",
        "fr": "Aucune donnée de maillage disponible",
        "it": "Nessun dato mesh disponibile",
    },
    "empty_bbox": {
        "de": "Bounding-Box leer",
        "fr": "Boîte englobante vide",
        "it": "Riquadro di delimitazione vuoto",
    },
    "loading_threejs": {
        "de": "Lade Three.js…",
        "fr": "Chargement de Three.js…",
        "it": "Caricamento Three.js…",
    },
    "building_scene": {"de": "Baue Szene…", "fr": "Construction de la scène…", "it": "Costruzione della scena…"},
    "distance_label": {"de": "Distanz", "fr": "Distance", "it": "Distanza"},
    "legend_no_rule": {"de": "Keine Regel", "fr": "Aucune règle", "it": "Nessuna regola"},
    "error_prefix": {"de": "Fehler", "fr": "Erreur", "it": "Errore"},
    "control_hint": {
        "de": "Maus: Drehen · Scroll: Zoom · Rechtsklick: Pan",
        "fr": "Souris : rotation · Molette : zoom · Clic droit : déplacer",
        "it": "Mouse: ruota · Scroll: zoom · Clic destro: sposta",
    },

    # ── Common measurements (metric labels used by viewer) ──
    "m_volume_m3": {"de": "Volumen (m³)", "fr": "Volume (m³)", "it": "Volume (m³)"},
    "m_surface_m2": {"de": "Oberfläche (m²)", "fr": "Surface (m²)", "it": "Superficie (m²)"},
    "m_watertight": {"de": "Wasserdicht", "fr": "Étanche", "it": "Stagno"},
    "m_crown_width_mm": {
        "de": "Kronenbreite (mm)",
        "fr": "Largeur couronn. (mm)",
        "it": "Larghezza corona (mm)",
    },
    "m_crown_slope_pct": {
        "de": "Kronenneigung (%)",
        "fr": "Pente couronn. (%)",
        "it": "Pendenza corona (%)",
    },
    "m_min_thickness_mm": {
        "de": "Wandstärke min (mm)",
        "fr": "Épaisseur min (mm)",
        "it": "Spessore min (mm)",
    },
    "m_wall_height_m": {"de": "Wandhöhe (m)", "fr": "Hauteur mur (m)", "it": "Altezza muro (m)"},
    "m_inclination_ratio": {"de": "Anzug (n:1)", "fr": "Fruit (n:1)", "it": "Rastremazione (n:1)"},
    "m_min_radius_m": {"de": "Min. Radius (m)", "fr": "Rayon min (m)", "it": "Raggio min (m)"},
    "m_plumbness_deg": {
        "de": "Lotabweichung (°)",
        "fr": "Écart d'aplomb (°)",
        "it": "Scostamento dal piombo (°)",
    },

    # ── Errors / warnings ────────────────────────────────
    "err_ifc_load": {
        "de": "IFC-Datei konnte nicht geladen werden",
        "fr": "Impossible de charger le fichier IFC",
        "it": "Impossibile caricare il file IFC",
    },
    "err_mesh_extract": {
        "de": "Geometrie-Extraktion fehlgeschlagen",
        "fr": "Échec de l'extraction de la géométrie",
        "it": "Estrazione della geometria fallita",
    },
    "err_no_elements": {
        "de": "Keine Elemente für Typen",
        "fr": "Aucun élément pour les types",
        "it": "Nessun elemento per i tipi",
    },
    "err_slope_skipped": {
        "de": "Kronen-Gefälle-Analyse übersprungen",
        "fr": "Analyse de pente de couronn. ignorée",
        "it": "Analisi pendenza corona saltata",
    },
    "err_anomaly_skipped": {
        "de": "Anomaly-Erkennung übersprungen",
        "fr": "Détection d'anomalies ignorée",
        "it": "Rilevamento anomalie saltato",
    },
    "err_html_failed": {
        "de": "HTML-Prüfprotokoll konnte nicht erstellt werden",
        "fr": "Impossible de générer le rapport HTML",
        "it": "Impossibile generare il rapporto HTML",
    },
    "err_csv_failed": {
        "de": "CSV-Export fehlgeschlagen",
        "fr": "Échec de l'export CSV",
        "it": "Esportazione CSV fallita",
    },
    "err_viewer_failed": {
        "de": "3D-Viewer konnte nicht geladen werden",
        "fr": "Impossible de charger la visionneuse 3D",
        "it": "Impossibile caricare il visualizzatore 3D",
    },
    "dl_html_report": {
        "de": "HTML-Prüfprotokoll herunterladen",
        "fr": "Télécharger le rapport HTML",
        "it": "Scarica rapporto HTML",
    },
    "dl_csv_export": {
        "de": "CSV-Export herunterladen",
        "fr": "Télécharger export CSV",
        "it": "Scarica esportazione CSV",
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
