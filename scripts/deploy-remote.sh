#!/bin/bash
set -euo pipefail

# ============================================
# Sunucu tarafi deploy script
# Kullanim: Bu script sunucuda calisir
# ============================================

PROJECT_DIR="${PROJECT_DIR:-/opt/moodle-student-tracker}"
SERVICE_NAME="${SERVICE_NAME:-telegram-bot}"
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"
BRANCH="${BRANCH:-main}"
LOG_FILE="${PROJECT_DIR}/logs/deploy_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
fail() { log "HATA: $1"; exit 1; }

mkdir -p "$(dirname "$LOG_FILE")"

log "========== Deploy basliyor =========="

# 1. Git pull
cd "$PROJECT_DIR" || fail "Proje dizini bulunamadi: $PROJECT_DIR"
log "Git pull ($BRANCH)..."
git fetch origin "$BRANCH" || fail "git fetch basarisiz"
git reset --hard "origin/$BRANCH" || fail "git reset basarisiz"
log "Git pull tamamlandi: $(git log --oneline -1)"

# 2. Dependency guncelleme
log "Dependency guncelleniyor..."
"$PIP" install -r requirements.txt --quiet || fail "pip install basarisiz"

# 3. Pre-deploy kontrol (basit import test)
log "Import kontrolu..."
"$PYTHON" -c "from bot.main import *" || fail "Import hatasi - deploy iptal"

# 4. Servisi yeniden baslat
log "Servis yeniden baslatiliyor..."
sudo systemctl restart "$SERVICE_NAME" || fail "Servis restart basarisiz"

# 5. Servis saglik kontrolu
sleep 5
if systemctl is-active --quiet "$SERVICE_NAME"; then
    log "Servis calisiyor"
else
    log "UYARI: Servis aktif degil!"
    sudo journalctl -u "$SERVICE_NAME" --no-pager -n 20 | tee -a "$LOG_FILE"
    fail "Servis baslatilamadi"
fi

log "========== Deploy tamamlandi =========="
