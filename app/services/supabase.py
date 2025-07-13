import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get Supabase credentials from environment variables
url: str = os.environ.get("SUPABASE_URL") or "https://atdordzshxrtafbogpiw.supabase.co"
key: str = os.environ.get("SUPABASE_SERVICE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF0ZG9yZHpzaHhydGFmYm9ncGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAzMjk5NzcsImV4cCI6MjA2NTkwNTk3N30.VaFd0kZ63XCMY9Q1ScP9Km2N-j9Cioz4mM8haYylUNA"

# Create a single, reusable Supabase client instance
try:
    supabase: Client = create_client(url, key)
    print("Successfully connected to Supabase.")
except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    supabase: Client = None
