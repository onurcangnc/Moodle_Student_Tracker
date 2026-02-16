#!/bin/bash
set -euo pipefail

# ============================================
# Lokal deploy script
# Kullanim: ./scripts/deploy.sh
# ============================================

REMOTE_HOST="${REMOTE_HOST:-user@server-ip}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-/opt/moodle-bot/scripts/deploy-remote.sh}"
BRANCH="${BRANCH:-main}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
fail() { log "HATA: $1"; exit 1; }

log "========== Pre-deploy kontrolleri =========="

# 1. Lint
log "Lint calistiriliyor..."
ruff check . || fail "Lint hatalari var - once duzelt"
log "Lint gecti"

# 2. Test
log "Testler calistiriliyor..."
python -m pytest tests/unit/ -v --tb=short || fail "Testler fail - once duzelt"
log "Testler gecti"

# 3. Uncommitted degisiklik kontrolu
if [ -n "$(git status --porcelain)" ]; then
    fail "Commit edilmemis degisiklikler var - once commit et"
fi
log "Working tree temiz"

# 4. Push
log "Git push..."
git push origin "$BRANCH" || fail "Git push basarisiz"
log "Push tamamlandi"

log "========== Sunucuya deploy tetikleniyor =========="

# 5. SSH ile remote deploy
ssh "$REMOTE_HOST" "BRANCH=$BRANCH bash $REMOTE_SCRIPT" || fail "Remote deploy basarisiz"

log "========== Deploy tamamlandi =========="
