#!/bin/bash

# Deploy script for chaamo-backend to fly.io

echo "🚀 Deploying chaamo-backend to fly.io..."

# Check if fly CLI is installed
if ! command -v fly &> /dev/null; then
    echo "❌ Fly CLI is not installed. Please install it first:"
    echo "   curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if user is logged in to fly
if ! fly auth whoami &> /dev/null; then
    echo "❌ Not logged in to fly.io. Please run:"
    echo "   fly auth login"
    exit 1
fi

# Set environment variables if not already set
if [ -z "$SUPABASE_URL" ]; then
    echo "⚠️  SUPABASE_URL environment variable is not set"
    echo "   Please set it before deploying:"
    echo "   export SUPABASE_URL=your_supabase_url"
fi

if [ -z "$SUPABASE_SERVICE_KEY" ]; then
    echo "⚠️  SUPABASE_SERVICE_KEY environment variable is not set"
    echo "   Please set it before deploying:"
    echo "   export SUPABASE_SERVICE_KEY=your_supabase_service_key"
fi

# Deploy the application
echo "📦 Building and deploying..."
fly deploy

# Check deployment status
echo "🔍 Checking deployment status..."
fly status

echo "✅ Deployment complete!"
echo "🌐 Your API should be available at: https://chaamo-backend.fly.dev"
echo "📚 Swagger documentation: https://chaamo-backend.fly.dev/docs" 
