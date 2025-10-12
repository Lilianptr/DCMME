# try1
import asyncio
from pynorm import RxNormClient

async def test_single_ndc():
    """
    Test a single NDC code to verify the format and API connection
    """
    # Test NDC in 5-4-2 format (from your CSV: Labeler-Product-Package)
    test_ndc = "0781-1506-10"
    # 00002771227
    print(f"Testing NDC: {test_ndc}")
    print("="*60)
    
    async with RxNormClient() as client:
        # Check API health
        is_healthy = await client.check_health()
        print(f"API Health: {is_healthy}\n")
        
        try:
            # Find RXCUI from NDC
            print(f"Looking up NDC: {test_ndc}")
            rxcui_result = await client.find_rxcui_by_id(test_ndc, "NDC")
            
            print(f"Result type: {type(rxcui_result)}")
            print(f"Result: {rxcui_result}")
            
            # Extract RXCUI
            rxcui = None
            if isinstance(rxcui_result, str):
                rxcui = rxcui_result
            elif isinstance(rxcui_result, list) and len(rxcui_result) > 0:
                rxcui = rxcui_result[0]
            
            if not rxcui:
                print("❌ No RXCUI found for this NDC")
                return
            
            print(f"\n✓ Found RXCUI: {rxcui}")
            
            # Get all properties
            print("\nGetting drug properties...")
            properties = await client.get_all_properties(rxcui)
            
            if properties:
                print("\nDrug Properties:")
                for prop in properties:
                    prop_name = getattr(prop, 'propName', 'Unknown')
                    prop_value = getattr(prop, 'propValue', 'Unknown')
                    print(f"  {prop_name}: {prop_value}")
            
            # Get related info to find ingredients
            print("\nGetting related information...")
            related_info = await client.get_all_related_info(rxcui)
            
            ingredients = []
            if related_info and hasattr(related_info, 'allRelatedGroup'):
                print(f"Related info type: {type(related_info.allRelatedGroup)}")
                all_groups = related_info.allRelatedGroup
                if not isinstance(all_groups, list):
                    all_groups = [all_groups]
                
                for group in all_groups:
                    if hasattr(group, 'conceptGroup'):
                        concept_groups = group.conceptGroup
                        if not isinstance(concept_groups, list):
                            concept_groups = [concept_groups]
                        
                        for concept_group in concept_groups:
                            tty = getattr(concept_group, 'tty', None)
                            if tty == 'IN':  # Ingredient
                                props = getattr(concept_group, 'conceptProperties', [])
                                if not isinstance(props, list):
                                    props = [props]
                                
                                for prop in props:
                                    name = getattr(prop, 'name', None)
                                    if name:
                                        ingredients.append(name)
            
            if ingredients:
                print(f"\n✓ Found Ingredients: {', '.join(ingredients)}")
            else:
                print("\n❌ No ingredients found")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_single_ndc())

