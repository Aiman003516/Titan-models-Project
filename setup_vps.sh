#!/bin/bash
# =============================================================================
# Titan Agent — VPS Setup Script
# Run this as root on a fresh Ubuntu 24.04 VPS
# =============================================================================

set -e

echo "═══════════════════════════════════════════════════"
echo "  Titan Agent — VPS Setup"
echo "═══════════════════════════════════════════════════"

# ── System packages ───────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -y
apt-get install -y python3.12 python3.12-venv python3-pip git nginx certbot python3-certbot-nginx ufw

# ── Create titan user ─────────────────────────────────────────────────────────
echo "[2/6] Creating titan user..."
if ! id "titan" &>/dev/null; then
    useradd -m -s /bin/bash titan
fi

# ── Clone project ─────────────────────────────────────────────────────────────
echo "[3/6] Cloning project..."
PROJ_DIR="/home/titan/titan-agent"
if [ -d "$PROJ_DIR" ]; then
    cd "$PROJ_DIR"
    git pull
else
    git clone https://github.com/Aiman003516/Titan-models-Project.git "$PROJ_DIR"
fi
chown -R titan:titan /home/titan

# ── Python virtual environment ────────────────────────────────────────────────
echo "[4/6] Setting up Python environment..."
cd "$PROJ_DIR"
sudo -u titan python3.12 -m venv venv
sudo -u titan venv/bin/pip install --upgrade pip
sudo -u titan venv/bin/pip install -r requirements.txt

# ── systemd service ───────────────────────────────────────────────────────────
echo "[5/6] Creating systemd service..."
cat > /etc/systemd/system/titan-agent.service << 'EOF'
[Unit]
Description=Titan ERP Generator Agent
After=network.target

[Service]
Type=simple
User=titan
Group=titan
WorkingDirectory=/home/titan/titan-agent
ExecStart=/home/titan/titan-agent/venv/bin/python -m uvicorn titan.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable titan-agent

# ── Firewall ──────────────────────────────────────────────────────────────────
echo "[6/6] Configuring firewall..."
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8000/tcp
ufw --force enable

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup Complete!"
echo ""
echo "  Next steps:"
echo "  1. Create .env file:"
echo "     sudo -u titan nano /home/titan/titan-agent/.env"
echo "  2. Start the service:"
echo "     systemctl start titan-agent"
echo "  3. Check status:"
echo "     systemctl status titan-agent"
echo "  4. View logs:"
echo "     journalctl -u titan-agent -f"
echo "═══════════════════════════════════════════════════"
