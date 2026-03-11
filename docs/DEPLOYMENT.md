# Deployment

## Docker

```bash
# Build and run
docker build -t hybrid-rag-system .
docker-compose up -d

# View logs / stop
docker-compose logs -f
docker-compose down
```

Set API keys in a `.env` file:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
LOG_LEVEL=INFO
```

```bash
docker-compose --env-file .env up -d
```

---

## Cloud

**Google Cloud Run:**
```bash
gcloud run deploy hybrid-rag-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**AWS ECS:** Push to ECR → create task definition → deploy to ECS service → attach ALB.

**Azure Container Instances:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name rag-api \
  --image hybrid-rag-system \
  --dns-name-label rag-api \
  --ports 8000
```

---

## Production Checklist

- [ ] Set API keys and disable mock LLM
- [ ] Add HTTPS via reverse proxy (nginx)
- [ ] Add authentication and rate limiting
- [ ] Monitor `/health` and `/metrics`
- [ ] Set alerts: P95 latency > 200ms, error rate > 5%

---

## Resource Requirements (10K documents)

| Resource | Minimum |
|----------|---------|
| Memory | 4GB |
| CPU | 2 cores |
| Disk | 10GB |
