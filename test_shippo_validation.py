#!/usr/bin/env python3
"""
Test script for Shippo address validation - Broadway 1 case debugging
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.shippo import shippo_validate_address
from utils.logger import api_logger

async def _test_broadway_address_async():
    """Test the specific Broadway 1 address that was causing issues"""
    
    # The problematic address from the checkpoint
    test_address = {
        "name": "John Doe",
        "email": "john.doe@example.com", 
        "phone": "+1234567890",
        "street1": "1 Broadway",
        "street2": "",
        "city": "New York", 
        "state": "NY",
        "zip": "10007",
        "country": "US"
    }
    
    print("Testing Broadway 1 address validation with duplicate cleanup...")
    print(f"Address: {test_address}")
    
    try:
        result = await shippo_validate_address(test_address)
        print(f"✅ Success! Address processed: {getattr(result, 'object_id', 'N/A')}")
        print(f"   Street: {getattr(result, 'street1', 'N/A')}")
        print(f"   City: {getattr(result, 'city', 'N/A')}")
        print(f"   State: {getattr(result, 'state', 'N/A')}")
        print(f"   Zip: {getattr(result, 'zip', 'N/A')}")
        
        # Test validation results handling
        vr = getattr(result, "validation_results", None)
        if vr is not None:
            is_valid = bool(getattr(vr, "is_valid", False))
            messages = getattr(vr, "messages", []) or []
            print(f"   Valid: {is_valid}")
            if messages:
                print(f"   Messages: {[getattr(m, 'text', str(m)) for m in messages[:2]]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        
    print("\n" + "="*50)

async def _test_similar_addresses_async():
    """Test multiple similar addresses to verify duplicate handling"""
    
    addresses = [
        {
            "name": "Jane Smith",
            "email": "jane@example.com",
            "street1": "1 Broadway", 
            "city": "New York",
            "state": "NY", 
            "zip": "10007",
            "country": "US"
        },
        {
            "name": "Bob Wilson", 
            "email": "bob@test.com",
            "street1": "1 Broadway St",  # Slight variation
            "city": "New York",
            "state": "NY",
            "zip": "10007", 
            "country": "US"
        },
        {
            "name": "Alice Johnson",
            "email": "alice@demo.com", 
            "street1": "Broadway 1",  # Different format
            "city": "New York",
            "state": "NY",
            "zip": "10007",
            "country": "US"
        }
    ]
    
    print("Testing similar addresses for duplicate handling...")
    
    for i, addr in enumerate(addresses, 1):
        print(f"\nTest {i}: {addr['name']} - {addr['street1']}")
        try:
            result = await shippo_validate_address(addr)
            print(f"   ✅ Result: {getattr(result, 'object_id', 'N/A')}")
            print(f"   Street: {getattr(result, 'street1', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

async def main():
    """Run all tests"""
    print("Shippo Address Validation Test Suite")
    print("=" * 50)
    
    # Check environment
    api_key = os.environ.get("SHIPPO_API_KEY")
    if not api_key:
        print("❌ SHIPPO_API_KEY not set")
        return
        
    print(f"Using API Key: {'test' if 'test' in api_key.lower() else 'live'} mode")
    print()
    
    await test_broadway_address()
    await test_similar_addresses()
    
    print("\nTest suite completed!")

def test_broadway_address():
    asyncio.run(_test_broadway_address_async())


def test_similar_addresses():
    asyncio.run(_test_similar_addresses_async())


if __name__ == "__main__":
    asyncio.run(main())
