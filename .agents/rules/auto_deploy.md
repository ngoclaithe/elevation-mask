# Auto-Deploy After Commit Rule

Whenever changes in this repository are committed and pushed to `main`, the agent MUST automatically perform remote deployment via SSH over Tailscale to the production server.

## Server & Deploy Details
- **Host (Tailscale IP):** `100.104.138.69`
- **SSH User:** `root`
- **Working Directory:** `/home/ubuntu/elevation-mask`
- **Service Port:** `8787` (Docker host port) -> `8080` (container internal)

## Deployment Command
Run the following command after pushing to `main`:
```bash
ssh root@100.104.138.69 "cd /home/ubuntu/elevation-mask && git pull origin main && docker compose up -d --build"
```

## Verification Step
Verify that the service is running and healthy:
```bash
ssh root@100.104.138.69 "curl -s http://localhost:8787/health"
```
Ensure it returns `{"ok": true, ...}`.
