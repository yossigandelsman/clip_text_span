import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm
from utils.factory import create_model_and_transforms, get_tokenizer
from torchvision.datasets import ImageNet, ImageFolder
from utils.siglip.modeling_siglip import SiglipVisionModel
from utils.siglip.processing_siglip import SiglipProcessor

def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="google/siglip2-base-patch16-224",
        type=str,
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/mnt/data/yossi/ILSVRC2012/", type=str, help="dataset path"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def main(args):
    """Calculates the projected residual stream for a dataset."""
    model = SiglipVisionModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()
    processor = SiglipProcessor.from_pretrained(args.model)
    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )

    # Data:
    transform = lambda x: processor(images=x, return_tensors="pt")
    ds = ImageFolder(root=os.path.join(args.data_path, "val"), transform=transform)
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    attention_results = []
    representation_results = []
    mlp_results = []
    
    for i, (inputs, _) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(args.device)
            outputs = model(**inputs)
            representation_results.append(outputs.pooler_output)
    # with open(
    #     os.path.join(args.output_dir, f"{args.dataset}_attn_{args.model}.npy"), "wb"
    # ) as f:
    #     np.save(f, np.concatenate(attention_results, axis=0))
    # with open(
    #     os.path.join(args.output_dir, f"{args.dataset}_mlp_{args.model}.npy"), "wb"
    # ) as f:
    #     np.save(f, np.concatenate(mlp_results, axis=0))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
