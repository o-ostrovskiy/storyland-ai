#!/bin/bash
# Setup script for Langfuse evaluation pipeline

set -e

echo "========================================================"
echo "StoryLand AI - Langfuse Evaluation Pipeline Setup"
echo "========================================================"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "Please edit .env and add your Langfuse credentials:"
    echo "  - LANGFUSE_SECRET_KEY"
    echo "  - LANGFUSE_PUBLIC_KEY"
    echo "  - LANGFUSE_HOST"
    echo ""
    exit 1
fi

# Check if Langfuse credentials are set
source .env

if [ -z "$LANGFUSE_SECRET_KEY" ] || [ -z "$LANGFUSE_PUBLIC_KEY" ]; then
    echo "❌ Langfuse credentials not configured in .env"
    echo ""
    echo "Please add the following to your .env file:"
    echo "  LANGFUSE_SECRET_KEY=sk-lf-your-secret-key"
    echo "  LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key"
    echo "  LANGFUSE_HOST=https://cloud.langfuse.com"
    echo ""
    echo "Get your API keys from: https://cloud.langfuse.com"
    exit 1
fi

echo "✅ Langfuse credentials configured"
echo ""

# Test authentication
echo "🔐 Testing Langfuse authentication..."
python3 -c "
from langfuse import Langfuse
try:
    client = Langfuse()
    if client.auth_check():
        print('✅ Authentication successful')
    else:
        print('❌ Authentication failed')
        exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Authentication failed. Check your credentials."
    exit 1
fi

echo ""
echo "📊 Creating Langfuse datasets from evalsets..."
python3 evaluation/tools/langfuse_eval.py --create-datasets --evalset-dir evaluation

echo ""
echo "========================================================"
echo "✅ Setup Complete!"
echo "========================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. View datasets in Langfuse:"
echo "   https://cloud.langfuse.com"
echo ""
echo "2. Run evaluations:"
echo "   make eval-run"
echo ""
echo "3. View results:"
echo "   make eval-summary"
echo "   make eval-report"
echo ""
echo "4. Set up automated evaluations:"
echo "   Add GitHub secrets and enable the workflow in:"
echo "   .github/workflows/scheduled-eval.yml"
echo ""
echo "📖 Full documentation: evaluation/README.md"
echo ""
