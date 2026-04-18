# Bella Notte Agent — Cloud Server Setup Guide

Complete setup instructions for deploying on a cloud GPU server (AWS, GCP, Azure).

---

## Prerequisites

- Ubuntu 22.04+ (or similar Linux)
- NVIDIA GPU (for PersonaPlex voice model)
- NVIDIA drivers + CUDA 12.x installed
- Python 3.11+
- At least 16GB GPU VRAM (for PersonaPlex 7B model)
- Google Gemini API key

---

## Step 1: System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and essentials
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Install audio libraries (needed for sounddevice)
sudo apt install -y portaudio19-dev libsndfile1

# Verify NVIDIA GPU
nvidia-smi
```

---

## Step 2: Clone the Repo

```bash
git clone https://github.com/bns218/Booking_System_voice.git
cd Booking_System_voice
```

---

## Step 3: Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 4: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit with your values
nano .env
```

**Required: Set your Gemini API key:**
```
GOOGLE_API_KEY=your-actual-gemini-api-key
```

Get a key at: https://aistudio.google.com/apikey

---

## Step 5: Add RAG Documents

Place your restaurant documents in the `docs/` folder:

```bash
ls docs/
# menu.pdf  wine_list.pdf  booking_policy.txt  faq.docx
```

These get automatically indexed into the vector store on server startup.

---

## Step 6: Start PersonaPlex (Voice Model)

PersonaPlex needs to run first — it provides the speech-to-speech AI:

```bash
# Generate SSL certificates (PersonaPlex requires SSL)
SSL_DIR=$(mktemp -d)

# Start PersonaPlex server on port 8998
python -m moshi.server \
    --ssl "$SSL_DIR" \
    --hf-repo nvidia/personaplex-7b-v1 \
    --port 8998
```

Wait until you see `Server started` before continuing. This downloads the model on first run (~14GB).

---

## Step 7: Start the FastAPI Server

In a new terminal (or use `tmux`/`screen`):

```bash
cd Booking_System_voice
source venv/bin/activate

# Start the backend server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
=== Bella Notte Server Starting ===
[OK] Database initialised
[OK] RAG context ready (X,XXX chars)
[OK] Persona prompt built (X,XXX chars)
=== Server ready — open http://localhost:8000 ===
```

---

## Step 8: Open the Test Console

Open your browser and go to:

```
http://YOUR_SERVER_IP:8000
```

The test console will auto-check if the server is connected.

### Two Testing Modes:

**Live Voice (default):**
- Click the green call button
- Allow microphone access
- Speak naturally — PersonaPlex handles speech-to-speech
- Transcript appears in real-time
- Booking details are auto-extracted by Gemini

**Text Simulation (click "Live Voice" badge to toggle):**
- Type messages in the input box
- Gemini responds as Sofia using the full persona + RAG
- No GPU needed for this mode
- Same booking extraction pipeline

---

## Running in Production

### Option A: Using tmux (simple)

```bash
# Start a tmux session
tmux new -s bella-notte

# Pane 1: PersonaPlex
python -m moshi.server --ssl "$SSL_DIR" --hf-repo nvidia/personaplex-7b-v1

# Split pane (Ctrl+B then %)
# Pane 2: FastAPI
uvicorn server:app --host 0.0.0.0 --port 8000

# Detach: Ctrl+B then D
# Reattach: tmux attach -t bella-notte
```

### Option B: Using systemd (production)

Create `/etc/systemd/system/bella-notte.service`:

```ini
[Unit]
Description=Bella Notte Booking Agent
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Booking_System_voice
Environment=PATH=/home/ubuntu/Booking_System_voice/venv/bin
ExecStart=/home/ubuntu/Booking_System_voice/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bella-notte
sudo systemctl start bella-notte
sudo systemctl status bella-notte
```

### Option C: Using Nginx reverse proxy (HTTPS)

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
sudo systemctl restart nginx
```

---

## Firewall / Security Group

Open these ports on your cloud provider:

| Port | Purpose |
|------|---------|
| 8000 | FastAPI server (or 443 if using Nginx) |
| 22   | SSH access |

PersonaPlex (8998) does NOT need to be exposed — the FastAPI server proxies to it internally.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Test console UI |
| GET | `/api/status` | Server health check |
| GET | `/api/bookings` | List all bookings |
| GET | `/api/bookings?date=Saturday` | Filter bookings by date |
| GET | `/api/bookings/{reference}` | Get single booking |
| POST | `/api/bookings/save` | Save a booking (JSON body) |
| GET | `/api/sessions` | List active call sessions |
| WS | `/ws/call` | Full voice pipeline WebSocket |
| WS | `/ws/simulate` | Text simulation WebSocket |

---

## Troubleshooting

**"Server offline" in the browser:**
- Make sure uvicorn is running on port 8000
- Check firewall allows port 8000
- If accessing remotely, update the Server URL in the config panel to `ws://YOUR_IP:8000/ws/call`

**PersonaPlex connection failed:**
- Verify PersonaPlex is running: `curl -k https://localhost:8998`
- Check GPU memory: `nvidia-smi`
- Make sure SSL certs were generated

**Microphone not working:**
- Browser requires HTTPS for microphone access (except localhost)
- Set up Nginx with SSL for remote access
- Check browser permissions

**RAG context empty:**
- Place documents in `docs/` folder
- Supported formats: .txt, .pdf, .docx
- Restart the server after adding docs

**Booking not extracting:**
- Verify GOOGLE_API_KEY is set in .env
- Check server logs for Gemini API errors
