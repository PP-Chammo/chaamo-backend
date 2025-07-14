#!/usr/bin/env python3
"""
Test script to debug Supabase upsert functionality
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.supabase import supabase
from app.models.card import SoldCard, FinalGroupedCard


async def test_supabase_connection():
    """Test basic Supabase connection and table access"""
    print("🔍 Testing Supabase connection...")
    
    if not supabase:
        print("❌ ERROR: Supabase client is None")
        return False
    
    try:
        # Test basic table access
        print("📊 Testing table access...")
        response = supabase.table('cards').select('id, title').limit(5).execute()
        print(f"✅ Table access successful")
        print(f"   📊 Found {len(response.data)} existing cards")
        
        if response.data:
            print("   📋 Sample cards:")
            for i, card in enumerate(response.data[:3], 1):
                print(f"      {i}. {card.get('title', 'No title')} (ID: {card.get('id', 'No ID')})")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR testing table access: {e}")
        return False


async def test_upsert_rpc():
    """Test the upsert_cards RPC function"""
    print("\n🔄 Testing upsert_cards RPC function...")
    
    if not supabase:
        print("❌ ERROR: Supabase client is None")
        return False
    
    # Create test data
    test_sold_card = SoldCard(
        item_id="test_item_123",
        sold_date="2024-01-15",
        price=25.99,
        currency="USD",
        link_url="https://example.com/test-card"
    )
    
    test_card_group = FinalGroupedCard(
        title="Test Card - Debug",
        image_url="https://example.com/test-image.jpg",
        sold_cards=[test_sold_card]
    )
    
    try:
        # Prepare parameters
        sold_cards_json = [sc.model_dump() for sc in test_card_group.sold_cards]
        
        params_to_send = {
            'card_title': test_card_group.title,
            'card_image_url': test_card_group.image_url,
            'card_sales_data': sold_cards_json
        }
        
        print(f"📤 Sending test data to upsert_cards RPC:")
        print(f"   - card_title: '{test_card_group.title}'")
        print(f"   - card_image_url: {test_card_group.image_url}")
        print(f"   - card_sales_data: {len(sold_cards_json)} items")
        
        # Call the RPC function
        response = supabase.rpc('upsert_cards', params_to_send).execute()
        
        print("✅ RPC call completed")
        
        # Analyze response
        if hasattr(response, 'data'):
            print(f"📋 Response data: {response.data}")
        else:
            print(f"📋 Response data: None")
            
        if hasattr(response, 'error') and response.error is not None:
            print(f"❌ ERROR in RPC response: {response.error}")
            if hasattr(response.error, 'message'):
                print(f"   Error message: {response.error.message}")
            if hasattr(response.error, 'details'):
                print(f"   Error details: {response.error.details}")
            return False
        else:
            print("✅ RPC call successful - no errors in response")
            return True
            
    except Exception as e:
        print(f"❌ EXCEPTION during RPC test: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


async def check_existing_data():
    """Check if the test data was actually saved"""
    print("\n🔍 Checking if test data was saved...")
    
    if not supabase:
        print("❌ ERROR: Supabase client is None")
        return False
    
    try:
        # Look for our test card
        response = supabase.table('cards').select('*').eq('title', 'Test Card - Debug').execute()
        
        print(f"📊 Found {len(response.data)} cards with test title")
        
        if response.data:
            print("✅ Test data was successfully saved!")
            for card in response.data:
                print(f"   📋 Card: {card.get('title')} (ID: {card.get('id')})")
                print(f"      Image: {card.get('image_url')}")
                print(f"      Sold cards: {len(card.get('sold_cards', []))} items")
        else:
            print("❌ Test data was not found in database")
            
        return len(response.data) > 0
        
    except Exception as e:
        print(f"❌ ERROR checking existing data: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 Starting Supabase upsert debug tests...\n")
    
    # Test 1: Connection
    connection_ok = await test_supabase_connection()
    if not connection_ok:
        print("❌ Connection test failed - stopping tests")
        return
    
    # Test 2: RPC function
    rpc_ok = await test_upsert_rpc()
    if not rpc_ok:
        print("❌ RPC test failed")
    
    # Test 3: Check if data was saved
    data_saved = await check_existing_data()
    
    print(f"\n🎯 Test Results:")
    print(f"   ✅ Connection: {'PASS' if connection_ok else 'FAIL'}")
    print(f"   ✅ RPC Function: {'PASS' if rpc_ok else 'FAIL'}")
    print(f"   ✅ Data Saved: {'PASS' if data_saved else 'FAIL'}")
    
    if connection_ok and rpc_ok and data_saved:
        print("🎉 All tests passed! Upsert functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main()) 
