#!/usr/bin/env bash
# Download and extract the Transperth static GTFS feed (no API key required).
# Usage:  bash prototype/download_data.sh
# Result: data/raw/google_transit.zip + data/raw/google_transit/*.txt
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$DIR/data/raw"
URL='https://www.transperth.wa.gov.au/LinkClick.aspx?link=%2fTimetablePDFs%2fGoogleTransit%2fProduction%2fgoogle_transit.zip&tabid=267&portalid=0&mid=1470'

mkdir -p "$RAW"
echo "Downloading Transperth GTFS -> $RAW/google_transit.zip"
curl -sSL -A 'Mozilla/5.0' -o "$RAW/google_transit.zip" "$URL"
ls -l "$RAW/google_transit.zip"
mkdir -p "$RAW/google_transit"
unzip -o -q "$RAW/google_transit.zip" -d "$RAW/google_transit"
echo "Extracted $(ls -1 "$RAW/google_transit" | wc -l) files to $RAW/google_transit"