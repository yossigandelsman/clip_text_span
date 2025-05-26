import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm
from utils.factory import create_model_and_transforms, get_tokenizer
from torchvision.datasets import ImageFolder
from utils.siglip.modeling_siglip import SiglipVisionModel, SiglipTextModel
from utils.siglip.processing_siglip import SiglipProcessor
from transformers import AutoTokenizer
from utils.openai_templates import OPENAI_IMAGENET_TEMPLATES
from utils.imagenet_classes import imagenet_classes
import torch.nn.functional as F
from typing import Union, Any


class ImageNet(ImageFolder):
    def __init__(self, root: Union[str, Path], split: str = "train", **kwargs: Any) -> None:
        wnid_to_classes = torch.load(os.path.join(root, "meta.bin"), weights_only=True)[0]
        super().__init__(os.path.join(root, split), **kwargs)

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}


def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="google/siglip-so400m-patch14-384",
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


def compute_zeroshot_weights(model, model_name, tokenizer, classnames, device, templates, use_format=False):
    max_length = {
        'google/siglip-so400m-patch14-384': 64,
        'google/siglip-base-patch16-224': 64
    }
    model.eval()
    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm.tqdm(classnames):
            texts = [template.format(c=classname) if use_format else template(classname) for template in templates]
            inputs = tokenizer(texts, truncation=False, padding="max_length", max_length=max_length[model_name], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            class_embedding = F.normalize(outputs.pooler_output, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding.cpu())
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    return zeroshot_weights


def main(args):
    """Calculates the projected residual stream for a dataset and zeroshot weights."""
    model = SiglipVisionModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()
    processor = SiglipProcessor.from_pretrained(args.model, use_fast=True)
    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )

    # Data:
    transform = lambda x: processor(images=x, return_tensors="pt")
    ds = ImageNet(root=args.data_path, split="val", transform=transform)
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    # Zeroshot weights for ImageNet
    print("Computing zeroshot weights for ImageNet...")
    text_model = SiglipTextModel.from_pretrained(args.model)
    text_model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    zeroshot_weights = compute_zeroshot_weights(text_model, args.model, tokenizer, imagenet_classes, args.device, templates=OPENAI_IMAGENET_TEMPLATES)
    np.save(os.path.join(args.output_dir, f"imagenet_zeroshot_weights_{args.model.replace('/', '_')}.npy"), zeroshot_weights.numpy())


    attention_results = []
    representation_results = []
    mlp_results = []
    
    for i, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(args.device)
            outputs = model(**inputs)
            representation_results.append(F.normalize(outputs.pooler_output, dim=-1))
            # Save labels for accuracy calculation
            if i == 0:
                all_labels = labels.clone()
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)


    # Compute accuracy
    print("Computing accuracy...")
    image_features = torch.cat(representation_results, dim=0)  # (N, D)
    zeroshot_weights = zeroshot_weights.to(image_features.device)  # (1000, D)
    logits = image_features @ zeroshot_weights.t()  # (N, 1000)
    preds = logits.argmax(dim=1)
    correct = (preds.cpu() == all_labels).sum().item()
    total = all_labels.size(0)
    acc = correct / total * 100
    print(f"Top-1 accuracy: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
