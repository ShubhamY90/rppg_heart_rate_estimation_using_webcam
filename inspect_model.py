import torch

ck = torch.load("UBFC-rPPG_RhythmFormer.pth", 
                map_location='cpu')

print("Type:", type(ck))

if isinstance(ck, dict):
    print("\nTop level keys:", list(ck.keys()))
    
    # Check if it has model weights
    for k, v in ck.items():
        if isinstance(v, dict):
            print(f"\n  '{k}' is a dict with {len(v)} keys")
            print(f"  First 5: {list(v.keys())[:5]}")
        elif hasattr(v, 'shape'):
            print(f"  '{k}' tensor shape: {v.shape}")
        else:
            print(f"  '{k}': {v}")