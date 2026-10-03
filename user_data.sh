#!/bin/bash
set -eux

# 1. Create 4GB swap (prevents OOM when loading models on 1GB RAM)
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# 2. System dependencies (OpenCV needs libgl1/libglib)
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip git nginx libgl1 libglib2.0-0

# 3. Clone application as ubuntu user
sudo -u ubuntu git clone https://github.com/myler71/Oral-Disease-YOLOxEfficientNet.git /home/ubuntu/app

# 4. Setup Python environment with CPU-only PyTorch first
sudo -u ubuntu python3 -m venv /home/ubuntu/app/.venv
sudo -u ubuntu /home/ubuntu/app/.venv/bin/pip install --upgrade pip
sudo -u ubuntu /home/ubuntu/app/.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
sudo -u ubuntu /home/ubuntu/app/.venv/bin/pip install -r /home/ubuntu/app/requirements.txt

# 5. Create systemd service
cat << 'EOF' > /etc/systemd/system/oral-streamlit.service
[Unit]
Description=Oral Disease Streamlit UI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
Environment=TF_CPP_MIN_LOG_LEVEL=2
Environment=YOLO_AUTOINSTALL=false
ExecStart=/home/ubuntu/app/.venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now oral-streamlit

# 6. Configure Nginx reverse proxy on port 80
cat << 'EOF' > /etc/nginx/sites-available/oral
server {
    listen 80;
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
EOF

ln -s /etc/nginx/sites-available/oral /etc/nginx/sites-enabled/oral
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
