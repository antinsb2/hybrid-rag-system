# Deployment Guide

Guide for deploying the Hybrid RAG System to production.

## Docker Deployment

### Build Image
```bash
docker build -t hybrid-rag-system .
```

### Run Container
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --name rag-api \
  hybrid-rag-system
```

### Using Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Environment Variables

Create `.env` file:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
LOG_LEVEL=INFO
```

Load with docker-compose:
```bash
docker-compose --env-file .env up -d
```

## Production Checklist

### Before Deployment

- [ ] Set production API keys
- [ ] Configure proper logging
- [ ] Set up monitoring
- [ ] Test with production data
- [ ] Configure rate limiting
- [ ] Set up backup strategy

### Security

- [ ] Use HTTPS (add reverse proxy like nginx)
- [ ] Add authentication
- [ ] Implement rate limiting
- [ ] Validate all inputs
- [ ] Sanitize file uploads
- [ ] Set resource limits

### Monitoring

- [ ] Monitor `/health` endpoint
- [ ] Track `/metrics` for anomalies
- [ ] Set up alerts for errors
- [ ] Monitor memory usage
- [ ] Track query latency P95/P99

## Cloud Deployment

### AWS ECS

1. Build and push to ECR
2. Create ECS task definition
3. Deploy to ECS service
4. Configure Application Load Balancer
5. Set up CloudWatch logging

### Google Cloud Run
```bash
gcloud run deploy hybrid-rag-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name rag-api \
  --image hybrid-rag-system \
  --dns-name-label rag-api \
  --ports 8000
```

## Scaling Considerations

### Horizontal Scaling

- Use shared vector store (Qdrant, Pinecone)
- Separate read replicas for queries
- Load balancer for API instances

### Vertical Scaling

- More memory for larger document sets
- GPU for faster embedding generation
- SSD for faster index loading

## Backup Strategy

### What to Backup

- Vector indexes (`/.cache/`)
- Embedding cache
- Configuration files
- Ingested documents metadata

### Backup Schedule

- Incremental: Daily
- Full: Weekly
- Test restores: Monthly

## Performance Optimization

### For Production
```python
# Use production-grade vector DB
pipeline = RAGPipeline(
    chunk_size=512,
    use_cache=True
)

# Enable all optimizations
pipeline.enable_hybrid(fusion_method="rrf")
```

### Resource Limits

Recommended for 10K documents:
- Memory: 4GB minimum
- CPU: 2 cores minimum
- Disk: 10GB minimum

## Monitoring Metrics

### Key Metrics to Track

- Request rate (requests/minute)
- Query latency (P50, P95, P99)
- Error rate (%)
- Cache hit rate (%)
- Memory usage
- CPU utilization

### Alert Thresholds

- P95 latency > 200ms
- Error rate > 5%
- Cache hit rate < 30%
- Memory usage > 90%
