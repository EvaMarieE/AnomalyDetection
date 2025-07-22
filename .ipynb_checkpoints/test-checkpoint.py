import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ldm.util import instantiate_from_config
#from models import UNET_models
from MedicalDataLoader import MedicalDataset

from models.UNET_models import UNetModel

def load_model(checkpoint_path, device):
    print(f"Loading model from checkpoint: {checkpoint_path}")

    model = UNetModel(
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512]
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_testing(args, model, device):
    print("Running on test set...")
    test_dataset = MedicalDataset(args, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs("test_results", exist_ok=True)

    for ii, batch in enumerate(test_loader):
        x = batch["image"].to(device)
        lungmask = batch.get("lungmask", None)
        if lungmask is not None:
            lungmask = lungmask.to(device)

        with torch.no_grad():
            output = model(x, mask=lungmask)

            # If model returns a dict (like in MAD-AD)
            pred_mask = output.get("anomaly_mask", output) if isinstance(output, dict) else output

            if pred_mask is not None:
                for i in range(x.size(0)):
                    plt.imsave(f"test_results/pred_{ii}_{i}.png", pred_mask[i][0].cpu().numpy(), cmap="gray")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 🔧 Load the model (no config or instantiate_from_config needed)
    model = load_model(args.checkpoint, args.device)

    # Optional: mock data config if MedicalDataset expects it
    args.data = {
        "image_size": 256,
        "root": "./data"
    }

    run_testing(args, model, args.device)
