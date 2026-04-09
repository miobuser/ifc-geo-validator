@echo off
title IFC Geometry Validator
echo ============================================================
echo   IFC Geometry Validator v2.0.0
echo   Geometrische Validierung von IFC-Infrastrukturmodellen
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python ist nicht installiert.
    echo Bitte installiere Python 3.10+ von https://www.python.org
    pause
    exit /b 1
)

:: Check if already installed
python -c "import ifc_geo_validator" >nul 2>&1
if errorlevel 1 (
    echo Erste Ausfuehrung: Installiere Abhaengigkeiten...
    echo Dies dauert 2-5 Minuten.
    echo.
    pip install -e ".[dev,bcf,web,viz]" -q
    if errorlevel 1 (
        echo FEHLER bei der Installation.
        pause
        exit /b 1
    )
    echo Installation abgeschlossen.
    echo.
)

echo Starte Web-App...
echo.
echo Die App oeffnet sich im Browser unter:
echo   http://localhost:8501
echo.
echo Zum Beenden: Ctrl+C in diesem Fenster
echo.
streamlit run src/ifc_geo_validator/app.py --server.headless false
