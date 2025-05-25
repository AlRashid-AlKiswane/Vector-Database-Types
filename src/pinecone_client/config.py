"""
Configuration file for Pinecone API key and default index settings.
"""

import sys
import os

# Get absolute path to the root of the project (VectorDatabase)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

# Load Pinecone API key from environment variable or fallback
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
    "pcsk_364yrb_GdSPGWu25oPSBYFLdzm4FnTD3tJqyxZLhrtcH8ii5qVpYP7QYmabHrCwsfVmgoa")

# Default index name
INDEX_NAME = "developer-quickstart-py"
