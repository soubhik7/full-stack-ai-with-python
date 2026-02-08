#!/usr/bin/env python3
"""
Fix SSL Certificate Verification for Python on macOS
"""
import ssl
import certifi
import os

# Set the SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Test SSL by fetching MNIST
from sklearn.datasets import fetch_openml

print("SSL certificate file location:", certifi.where())
print("\nAttempting to download MNIST dataset...")

try:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    print("✅ Success! MNIST dataset downloaded successfully!")
    print(f"Dataset shape: {mnist.data.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
