import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get Supabase credentials from environment variables
url: str = os.environ.get("SUPABASE_URL") or "https://atdordzshxrtafbogpiw.supabase.co"
key: str = os.environ.get("SUPABASE_SERVICE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF0ZG9yZHpzaHhydGFmYm9ncGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAzMjk5NzcsImV4cCI6MjA2NTkwNTk3N30.VaFd0kZ63XCMY9Q1ScP9Km2N-j9Cioz4mM8haYylUNA"

print(f"🔧 Initializing Supabase connection...")
print(f"   🌐 URL: {url}")
print(f"   🔑 Key: {key[:20]}..." if len(key) > 20 else f"   🔑 Key: {key}")

# Create a single, reusable Supabase client instance
try:
    supabase: Client = create_client(url, key)
    print("✅ Successfully created Supabase client")
    
    # Test the connection by making a simple query
    try:
        print("🔍 Testing Supabase connection...")
        test_response = supabase.table('cards').select('id').limit(1).execute()
        print("✅ Supabase connection test successful")
        print(f"   📊 Test query returned: {len(test_response.data)} rows")
    except Exception as test_error:
        print(f"⚠️  WARNING: Supabase connection test failed: {test_error}")
        print("   The client was created but may not be fully functional")
        
except Exception as e:
    print(f"❌ ERROR connecting to Supabase: {e}")
    print(f"   Error type: {type(e).__name__}")
    supabase: Client = None
