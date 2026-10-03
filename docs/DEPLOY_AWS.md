# Deploying to AWS (free tier)

This guide deploys the Streamlit UI (`streamlit_app.py`) to one EC2 instance. It does not deploy the FastAPI server. The UI loads both models in-process, so you only run one Python process.

## 1. Target architecture

```
Browser ──HTTP :80──▶ nginx ──▶ 127.0.0.1:8501 Streamlit (systemd: oral-streamlit)
                                   └─ EfficientNetB0 (TensorFlow) + YOLOv8n (PyTorch)
```

- **One EC2 instance**: Streamlit runs under systemd and nginx reverse-proxies port 80 to `127.0.0.1:8501`. nginx also handles the websocket upgrade Streamlit needs.
- **Why one process**: TensorFlow and PyTorch together use about 1–1.3 GB RSS once both models are loaded. Running FastAPI next to Streamlit would load both frameworks twice, which a 1 GiB free-tier instance can't hold. A single process plus a swap file fits on `t3.micro`.

## 2. Why not the alternatives

| Option | Why not |
|---|---|
| Lambda | Streamlit needs long-lived websockets. TF + torch images are huge, so cold starts take tens of seconds. |
| App Runner / ECS Fargate | No free tier; you pay per vCPU/GB-hour from the first minute. |
| Elastic Beanstalk | It's a wrapper around the same EC2 instance, with more moving parts (ASG, ELB if enabled) and nothing gained here. |
| SageMaker endpoints | Real-time endpoints aren't free-tier for continuous serving. |

## 3. Account and free-tier notes

Check which kind of account you have:

- **Free plan** (accounts created after July 2025): $100 in credits at sign-up, plus up to $100 more for completing onboarding activities. The account closes after 6 months or when the credits run out, whichever comes first. A `t3.micro` costs $0.0104/h in `us-east-1` (≈ $7.6/month), which is charged against the credits.
- **Legacy free tier** (older accounts): 750 h/month of `t2.micro`/`t3.micro` for 12 months, plus 30 GB of EBS.

Before launching anything:

1. **Create a zero-spend budget**: Billing and Cost Management → Budgets → Create budget → *Use a template* → **Zero spend budget**. You'll get an email as soon as anything costs money.
2. In the launch wizard, filter instance types by **Free tier eligible**.
3. **Public IPv4 is billed** at $0.005/h per address (≈ $3.6/month) unless your free tier or credits cover it. Check this under Free Tier usage after the first day.

## 4. Launch settings (EC2 → Launch instance)

| Setting | Value |
|---|---|
| Region | `us-east-1` |
| AMI | Ubuntu Server 24.04 LTS, 64-bit (x86) |
| Instance type | `t3.micro` (1 GiB). Upgrade to `t3.small` (2 GiB) if the OOM killer stops Streamlit (`journalctl -u oral-streamlit` shows `Killed`). |
| Key pair | Create or select one (needed for SSH) |
| Storage | 20 GiB gp3 (within the 30 GB free EBS allowance) |
| Security group | Inbound `22/tcp` from **My IP**; inbound `80/tcp` from `0.0.0.0/0` |

SSH in once the instance is running:

```bash
ssh -i /path/to/key.pem ubuntu@<public-ip>
```

## 5. Server setup

System packages (OpenCV, which ultralytics pulls in, needs `libgl1`/`libglib2.0-0`):

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git nginx libgl1 libglib2.0-0
```

4 GiB swap file (needed on a 1 GiB instance while the models load):

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

App and Python environment:

```bash
git clone https://github.com/myler71/Oral-Disease-YOLOxEfficientNet.git ~/app && cd ~/app && python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# CPU-only torch FIRST. Otherwise pip pulls multi-GB CUDA wheels via ultralytics.
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

## 6. systemd service

Create `/etc/systemd/system/oral-streamlit.service`:

```ini
[Unit]
Description=Oral Disease Streamlit UI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/app
Environment=TF_CPP_MIN_LOG_LEVEL=2
ExecStart=/home/ubuntu/app/.venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now oral-streamlit
sudo systemctl status oral-streamlit   # should show active (running)
```

## 7. nginx reverse proxy

Create `/etc/nginx/sites-available/oral`:

```nginx
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
```

`client_max_body_size 10M` matches `maxUploadSize = 10` in `.streamlit/config.toml`.

Enable it and remove the default site:

```bash
sudo ln -s /etc/nginx/sites-available/oral /etc/nginx/sites-enabled/oral
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

## 8. Optional: HTTPS

1. Get a free DNS name (for example [DuckDNS](https://www.duckdns.org/)) and point it at the instance's public IP.
2. Open inbound `443/tcp` from `0.0.0.0/0` in the security group.
3. Issue a certificate. Certbot rewrites the nginx site for you:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d <name>.duckdns.org
```

## 9. Verify

```bash
curl http://<public-ip>/_stcore/health   # → ok
```

Then open `http://<public-ip>/` in a browser. The sidebar should show two ✅ model lines. Upload an oral photo and check that the **YOLOv8n detection** and **EfficientNetB0 classification** panels both render. The first request is slower because the models load on first use and the instance may swap.

## 10. Update / redeploy

```bash
cd ~/app && git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart oral-streamlit
```

## 11. Cost hygiene

- **Stop** the instance when you aren't using it (EC2 → Instance state → Stop). A stopped instance doesn't bill compute. Its EBS volume still bills, but 20 GiB stays within the 30 GB free allowance.
- **Terminate** the instance when you're done for good, and check that the volume is deleted too.
- Check **Billing → Free Tier** usage regularly, and keep the zero-spend budget enabled.
