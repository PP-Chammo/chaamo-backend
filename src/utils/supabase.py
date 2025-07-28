import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Optional

load_dotenv()

url: Optional[str] = os.environ.get("SUPABASE_URL")
key: Optional[str] = os.environ.get("SUPABASE_SERVICE_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")

print("🔧 Initializing Supabase connection...")
print(f"   🌐 URL: {url}")
print(f"   🔑 Key: {key[:20]}..." if len(key) > 20 else f"   🔑 Key: {key}")

supabase: Optional[Client] = None

try:
    supabase = create_client(url, key)
    print("✅ Successfully created Supabase client")

    try:
        print("🔍 Testing Supabase connection...")
        test_response = supabase.table('categories').select('id').limit(1).execute()
        print("✅ Supabase connection test successful")
        print(f"   📊 Test query returned: {len(test_response.data)} rows")
    except Exception as test_error:
        print(f"⚠️  WARNING: Supabase connection test failed: {test_error}")
        print("   The client was created but may not be fully functional")

except Exception as e:
    print(f"❌ ERROR connecting to Supabase: {e}")
    print(f"   Error type: {type(e).__name__}")
