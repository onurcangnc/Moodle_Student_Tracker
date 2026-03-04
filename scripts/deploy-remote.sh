#!/bin/bash
set -euo pipefail

# ============================================
# Sunucu tarafi deploy script
# Kullanim: Bu script sunucuda calisir
# ============================================

PROJECT_DIR="${PROJECT_DIR:-/opt/moodle-bot}"
SERVICE_NAME="${SERVICE_NAME:-moodle-bot}"
BRANCH="${BRANCH:-main}"
LOG_FILE="${PROJECT_DIR}/logs/deploy_$(date +%Y%m%d_%H%M%S).log"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"

if [ -x "$VENV_DIR/bin/python" ] && [ -x "$VENV_DIR/bin/pip" ]; then
    PYTHON="${PYTHON:-$VENV_DIR/bin/python}"
    PIP="${PIP:-$VENV_DIR/bin/pip}"
else
    PYTHON="${PYTHON:-python3}"
    PIP="${PIP:-pip3}"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
fail() { log "HATA: $1"; exit 1; }

mkdir -p "$(dirname "$LOG_FILE")"

log "========== Deploy basliyor =========="

# 1. Git pull
cd "$PROJECT_DIR" || fail "Proje dizini bulunamadi: $PROJECT_DIR"
PREV_COMMIT="$(git rev-parse HEAD)"
log "Mevcut commit (rollback noktasi): $PREV_COMMIT"
log "Git pull ($BRANCH)..."
git fetch origin "$BRANCH" || fail "git fetch basarisiz"
git reset --hard "origin/$BRANCH" || fail "git reset basarisiz"
log "Git pull tamamlandi: $(git log --oneline -1)"

if [ "$PYTHON" = "python3" ] && [ "$PIP" = "pip3" ] && [ ! -d "$VENV_DIR" ]; then
    log "Venv bulunamadi, olusturuluyor: $VENV_DIR"
    python3 -m venv "$VENV_DIR" || fail "venv olusturma basarisiz"
    PYTHON="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
fi

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
    log "Servis process kontrolu PASS"
else
    log "UYARI: Servis process kontrolu FAIL, rollback tetikleniyor"
    git reset --hard "$PREV_COMMIT" || log "Rollback git reset basarisiz"
    "$PIP" install -r requirements.txt --quiet || log "Rollback pip install basarisiz"
    sudo systemctl restart "$SERVICE_NAME" || true
    sleep 5
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "Rollback basarili - $PREV_COMMIT commit'ine donuldu"
    else
        log "KRITIK: Rollback sonrasi servis baslatilamadi"
        sudo journalctl -u "$SERVICE_NAME" --no-pager -n 30 | tee -a "$LOG_FILE"
    fi
    exit 1
fi

log "Health check bekleniyor..."
HEALTH_OK=false
for i in $(seq 1 6); do
    sleep 5
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        HEALTH_OK=true
        HEALTH_BODY=$(curl -s http://localhost:9090/health)
        log "Health check PASS - $HEALTH_BODY"
        break
    fi
    log "Health check bekleniyor... ($i/6)"
done

if [ "$HEALTH_OK" = false ]; then
    log "UYARI: Health check 30 saniyede gecmedi - rollback tetikleniyor"
    git reset --hard "$PREV_COMMIT" || log "Rollback git reset basarisiz"
    "$PIP" install -r requirements.txt --quiet || log "Rollback pip install basarisiz"
    sudo systemctl restart "$SERVICE_NAME" || true
    sleep 5
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "Rollback basarili - $PREV_COMMIT commit'ine donuldu"
    else
        log "KRITIK: Rollback sonrasi servis baslatilamadi"
        sudo journalctl -u "$SERVICE_NAME" --no-pager -n 30 | tee -a "$LOG_FILE"
    fi
    exit 1
fi

log "========== Deploy tamamlandi =========="
