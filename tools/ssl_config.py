"""
SSL Configuration for Python on macOS

This script configures SSL certificates for Python to fix certificate verification errors
when downloading datasets or accessing HTTPS resources.

Add this to your notebook BEFORE importing sklearn.datasets or making HTTPS requests.
"""
import os
import certifi

# Configure SSL to use certifi's certificate bundle
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

print(f"✅ SSL configured. Using certificates from: {certifi.where()}")
