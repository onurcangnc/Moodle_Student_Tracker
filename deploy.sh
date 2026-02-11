#!/bin/bash
set -e

SERVER="root@46.37.115.3"
REMOTE_DIR="/opt/moodle-bot"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Moodle Bot Deployment ==="
echo "Server: $SERVER"
echo "Remote: $REMOTE_DIR"
echo ""

# 1. Sunucuda dizin oluştur ve Python hazırla
echo "[1/7] Sunucu hazırlanıyor..."
ssh $SERVER "
  apt-get update -qq && \
  apt-get install -y -qq python3 python3-venv python3-pip tesseract-ocr tesseract-ocr-tur tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-lat tesseract-ocr-ita tesseract-ocr-spa > /dev/null 2>&1 && \
  mkdir -p $REMOTE_DIR/data
"

# 2. Proje dosyalarını gönder (.env, venv, data, __pycache__ hariç)
echo "[2/7] Dosyalar gönderiliyor..."
rsync -avz --progress \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude 'data/' \
  --exclude '.env' \
  --exclude 'moodle-ai-assistant/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude '.claude/' \
  "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

# 3. .env dosyasını gönder
echo "[3/7] .env gönderiliyor..."
if [ -f "$LOCAL_DIR/.env" ]; then
  scp "$LOCAL_DIR/.env" "$SERVER:$REMOTE_DIR/.env"
  echo "  .env gönderildi."
else
  echo "  UYARI: Lokal .env bulunamadı! Sunucuda manuel oluşturmanız gerekecek."
fi

# 4. Sunucuda venv + pip install
echo "[4/7] Python bağımlılıkları kuruluyor..."
ssh $SERVER "
  cd $REMOTE_DIR && \
  python3 -m venv venv && \
  source venv/bin/activate && \
  pip install --upgrade pip -q && \
  pip install -r requirements.txt -q
"

# 5. Systemd servisi kur
echo "[5/7] Systemd servisi kuruluyor..."
scp "$LOCAL_DIR/moodle-bot.service" "$SERVER:/etc/systemd/system/moodle-bot.service"
ssh $SERVER "
  systemctl daemon-reload && \
  systemctl enable moodle-bot
"

# 6. HuggingFace embedding modelini indir
echo "[6/7] Embedding modeli indiriliyor (all-MiniLM-L6-v2)..."
ssh $SERVER "
  cd $REMOTE_DIR && \
  source venv/bin/activate && \
  python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model başarıyla indirildi.')\"
"

# Model indirildikten sonra HF_HUB_OFFLINE=1 yap (cache'den yükle)
echo "  HF_HUB_OFFLINE=1 olarak ayarlanıyor..."
ssh $SERVER "sed -i 's/^HF_HUB_OFFLINE=0/HF_HUB_OFFLINE=1/' $REMOTE_DIR/.env"

# Servisi başlat
ssh $SERVER "systemctl restart moodle-bot"

# 7. Durum kontrolü
echo "[7/7] Servis durumu kontrol ediliyor..."
ssh $SERVER "systemctl status moodle-bot --no-pager"

echo ""
echo "=== Deployment tamamlandı! ==="
echo "Log takibi: ssh $SERVER 'journalctl -u moodle-bot -f'"
