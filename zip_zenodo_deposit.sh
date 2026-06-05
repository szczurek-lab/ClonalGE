#!/usr/bin/env bash
# Wait for the build script to finish, then zip the deposit.
set -uo pipefail

PID_FILE="/Volumes/LenovoPS8/zenodo_build.pid"
DEST="/Volumes/LenovoPS8/zenodo_clonalge"
ZIP="/Volumes/LenovoPS8/clonalge_zenodo.zip"
LOG="/Volumes/LenovoPS8/zenodo_zip.log"

if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found: $PID_FILE" | tee -a "$LOG"
    exit 1
fi

PID=$(cat "$PID_FILE")
echo "[$(date)] Waiting for build PID $PID to finish..." | tee -a "$LOG"

# Poll every 30s until process is gone
while kill -0 "$PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Build finished. Verifying deposit..." | tee -a "$LOG"

if [ ! -d "$DEST" ]; then
    echo "ERROR: deposit folder $DEST does not exist." | tee -a "$LOG"
    exit 1
fi

# Sanity: deposit should have at least the README and 5 numbered subfolders
if [ ! -f "$DEST/README.md" ]; then
    echo "ERROR: $DEST/README.md missing — build likely failed" | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Deposit size:" | tee -a "$LOG"
du -sh "$DEST" | tee -a "$LOG"

echo "[$(date)] Creating zip → $ZIP" | tee -a "$LOG"
cd /Volumes/LenovoPS8
# -r recursive, -y store symlinks as-is, -X no extra Mac attrs,
# --exclude AppleDouble + Finder metadata
zip -r -y -X "$ZIP" zenodo_clonalge \
    --exclude '*/._*' --exclude '*/.DS_Store' \
    >> "$LOG" 2>&1

echo "[$(date)] Zip done. Size:" | tee -a "$LOG"
du -sh "$ZIP" | tee -a "$LOG"

echo "[$(date)] All steps complete." | tee -a "$LOG"
