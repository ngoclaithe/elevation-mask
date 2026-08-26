# Elevation Mask - Agent Instructions

## Deployment Workflow
Sau mỗi lần commit & push lên branch `main`, hãy tự động SSH vào server qua Tailscale IP và tiến hành pull code mới và rebuild docker container:

```bash
ssh root@100.104.138.69 "cd /home/ubuntu/elevation-mask && git pull origin main && docker compose up -d --build"
```

Sau đó kiểm tra health check:
```bash
ssh root@100.104.138.69 "curl -s http://localhost:8787/health"
```
Server thông tin:
- **Tailscale IP:** `100.104.138.69`
- **User:** `root`
- **Directory:** `/home/ubuntu/elevation-mask`
- **Host port:** `8787`
