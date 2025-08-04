# 🚀 Chaamo Backend Deployment Guide

This guide will help you deploy the chaamo-backend FastAPI application to fly.io.

## 📋 Prerequisites

1. **Fly CLI**: Install the Fly CLI
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Fly Account**: Sign up and log in to fly.io
   ```bash
   fly auth login
   ```

3. **Supabase Setup**: Ensure you have:
   - Supabase project URL
   - Supabase service key (not anon key)

## 🔧 Environment Variables

The application requires the following environment variables:

```bash
export SUPABASE_URL="https://your-project-ref.supabase.co"
export SUPABASE_SERVICE_KEY="your_service_key_here"
```

## 🐳 Docker Build

The application uses a multi-stage Docker build with:

- **Base Image**: Python 3.11-slim
- **Dependencies**: All system libraries for Playwright
- **Security**: Non-root user execution
- **Health Checks**: Built-in health monitoring

## 🚀 Deployment Steps

### Option 1: Using the deployment script

```bash
# Make sure you're in the chaamo-backend directory
cd chaamo-backend

# Set environment variables
export SUPABASE_URL="your_supabase_url"
export SUPABASE_SERVICE_KEY="your_service_key"

# Run the deployment script
./deploy.sh
```

### Option 2: Manual deployment

```bash
# Set environment variables
fly secrets set SUPABASE_URL="your_supabase_url"
fly secrets set SUPABASE_SERVICE_KEY="your_service_key"

# Deploy
fly deploy
```

## 🌐 Application Endpoints

Once deployed, your application will be available at:

- **Main API**: `https://chaamo-backend.fly.dev`
- **Swagger Documentation**: `https://chaamo-backend.fly.dev/docs`
- **ReDoc Documentation**: `https://chaamo-backend.fly.dev/redoc`
- **Health Check**: `https://chaamo-backend.fly.dev/`

## 📚 API Endpoints

### Available Endpoints

1. **GET /** - Health check and welcome message
2. **GET /docs** - Swagger UI documentation
3. **GET /api/v1/ebay_search** - eBay scraping endpoint

### eBay Search Parameters

- `query` (string): Search query (e.g., "2024 Topps Thiery Henry")
- `region` (enum): "us" or "uk"
- `master_card_id` (optional): Master card ID from database

## 🔍 Monitoring

### Health Checks

The application includes built-in health checks that monitor:
- Application responsiveness
- Database connectivity
- Playwright browser availability

### Logs

View application logs:
```bash
fly logs
```

### Status

Check deployment status:
```bash
fly status
```

## 🛠️ Troubleshooting

### Common Issues

1. **Playwright Installation**: The Dockerfile includes all necessary dependencies for Playwright
2. **Memory Issues**: The application is configured to run with minimal memory usage
3. **Database Connection**: Ensure Supabase credentials are correctly set

### Debug Commands

```bash
# Check application logs
fly logs

# SSH into the running container
fly ssh console

# Check application status
fly status

# Restart the application
fly apps restart chaamo-backend
```

## 🔒 Security

- Non-root user execution
- HTTPS enforcement
- Environment variable secrets management
- CORS properly configured

## 📈 Scaling

The application is configured for:
- Auto-scaling based on demand
- Zero-downtime deployments
- Health check-based restarts

## 🆘 Support

If you encounter issues:

1. Check the logs: `fly logs`
2. Verify environment variables: `fly secrets list`
3. Restart the application: `fly apps restart chaamo-backend` 
