import torch

def inspect(obj, indent=0):
    prefix = " " * indent

    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}: {type(v)}")
            inspect(v, indent + 4)

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} (len={len(obj)})")
        for item in obj:
            inspect(item, indent + 4)

    elif torch.is_tensor(obj):
        print(f"{prefix}Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}")

    else:
        print(f"{prefix}{repr(obj)}")

# data = torch.load("./results/checkpoints/hybrid_vae/hybrid_vae_latent_2.pt", map_location="cpu")
# data = torch.load("./results/checkpoints/hybrid_vae/hybrid_vae_latent_4.pt", map_location="cpu")
# data = torch.load("./results/checkpoints/hybrid_vae/hybrid_vae_latent_8.pt", map_location="cpu")
data = torch.load("./results/checkpoints/hybrid_vae/hybrid_vae_latent_16.pt", map_location="cpu")
inspect(data)