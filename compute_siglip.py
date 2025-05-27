import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
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
from compute_complete_text_set import replace_with_iterative_removal

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
    parser.add_argument("--save_everything", action="store_true", help="save everything") 
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--compute_text_spans", action="store_true", help="compute text spans")
    parser.add_argument("--text_descriptions", default="text_descriptions/image_descriptions_general.txt", type=str, help="text descriptions to use")
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

@torch.no_grad()
def get_text_features(model, model_name, tokenizer, lines, 
                      device, batch_size):
    max_length = {
        'google/siglip-so400m-patch14-384': 64,
        'google/siglip-base-patch16-224': 64
    }
    model.eval()
    zeroshot_weights = []
    for i in tqdm.trange(0, len(lines), batch_size):
        texts = [l.replace('\n', '') for l in lines[i:i+batch_size]]
        inputs = tokenizer(texts, truncation=False, padding="max_length", max_length=max_length[model_name], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        class_embedding = F.normalize(outputs.pooler_output, dim=-1)
        zeroshot_weights.append(class_embedding.detach().cpu())
    zeroshot_weights = torch.concatenate(zeroshot_weights, dim=0)
    return zeroshot_weights


# Minimal PRS hook for out.post and mlp_output
class PRSHook:
    def __init__(self, collapse_spatial: bool = False):
        self.attention_records = []
        self.mlp_records = []
        self.collapse_spatial = collapse_spatial
    
    def save_attention(self, ret, **kwargs):
        if self.collapse_spatial:
            to_return = ret.sum(axis=2).detach().cpu()
            self.attention_records.append(to_return)
        else:
            self.attention_records.append(ret.detach().cpu())
        return ret
    
    def save_mlp(self, ret, **kwargs):
        self.mlp_records.append(ret.detach().cpu())
        return ret
    
    def finalize(self):
        self.attention_records = torch.cat(self.attention_records, dim=0)
        self.mlp_records = torch.cat(self.mlp_records, dim=0)
        return {"attention_records": self.attention_records, "mlp_records": self.mlp_records}

def compute_accuracy(features, labels, zeroshot_weights):
    zeroshot_weights = zeroshot_weights.to(features.device)  # (1000, D)
    logits = features @ zeroshot_weights.t()  # (N, 1000)
    preds = logits.argmax(dim=1)
    correct = (preds.cpu() == labels).sum().item()
    total = labels.size(0) 
    acc = correct / total * 100
    return acc, correct, total

@torch.no_grad()
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
    zeroshot_weights_path = os.path.join(args.output_dir, f"imagenet_zeroshot_weights_{args.model.replace('/', '_')}.npy")
    if os.path.exists(zeroshot_weights_path):
        print(f"Loading zeroshot weights from {zeroshot_weights_path}")
        zeroshot_weights = torch.from_numpy(np.load(zeroshot_weights_path))
    else:
        zeroshot_weights = compute_zeroshot_weights(text_model, args.model, tokenizer, imagenet_classes, args.device, templates=OPENAI_IMAGENET_TEMPLATES)
        np.save(zeroshot_weights_path, zeroshot_weights.numpy())

    prs_hook = PRSHook(collapse_spatial=True)
    
    # # Register the hook on the model's hook manager
    model.hook.register('pooling_head.attention.out.post', prs_hook.save_attention)
    model.hook.register('pooling_head.mlp_output', prs_hook.save_mlp)
    
    # Compute representation accuracy
    representation_results = []
    for i, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(args.device)
        outputs = model(**inputs)
        representation_results.append(outputs.pooler_output)
        # Save labels for accuracy calculation
        if i == 0:
            all_labels = labels.clone()
        else:
            all_labels = torch.cat([all_labels, labels], dim=0)
    representation_results = torch.cat(representation_results, dim=0)  # (N, D)
    constant_bias = model.vision_model.head.attention.out_proj.bias.detach().cpu()
    # To be more percise, there is also a bias in the mlp output
    mlp_bias = model.vision_model.head.mlp.fc2.bias.detach().cpu()
    # Compute representation accuracy
    print("Computing accuracy (representation)...")
    acc, correct, total = compute_accuracy(representation_results, all_labels, zeroshot_weights)
    print(f"Top-1 representation accuracy: {acc:.2f}% ({correct}/{total})")

    attn_and_mlp_results = prs_hook.finalize()
    attn_results = attn_and_mlp_results["attention_records"]
    mlp_results = attn_and_mlp_results["mlp_records"]
    # Compute mlp accuracy
    print("Computing accuracy (mlp)...")
    acc, correct, total = compute_accuracy(mlp_results[:, 0] + constant_bias, all_labels, zeroshot_weights)
    print(f"Top-1 mlp accuracy: {acc:.2f}% ({correct}/{total})")
    
    # Compute attention accuracy
    print("Computing accuracy (attention)...")
    acc, correct, total = compute_accuracy(attn_results.sum(axis=2)[:, 0] + constant_bias + mlp_bias, all_labels, zeroshot_weights)
    print(f"Top-1 attention accuracy: {acc:.2f}% ({correct}/{total})")
    
    # Compute attention + mlp accuracy
    print("Computing accuracy (attention + mlp for sanity check)...")
    acc, correct, total = compute_accuracy(mlp_results[:, 0] + attn_results.sum(axis=2)[:, 0] + constant_bias, all_labels, zeroshot_weights)
    print(f"Top-1 attention + mlp accuracy: {acc:.2f}% ({correct}/{total})")
    
    # Optionally, save to disk:
    if args.save_everything:
        torch.save(attn_and_mlp_results, os.path.join(args.output_dir, f"siglip_{args.model.replace('/', '_')}_prs.pt"))
    
    # Compute text features
    with open(args.text_descriptions, 'r') as f:
            lines = f.readlines()
    base, name = os.path.split(args.text_descriptions)
    name = name.replace('.txt', '')
    text_features_path = os.path.join(args.output_dir, f'{name}_{args.model.replace('/', '_')}.npy')
    if os.path.exists(text_features_path):
        print(f"Loading text features from {text_features_path}")
        text_features = np.load(text_features_path)
    else:
        text_features = get_text_features(text_model, args.model, tokenizer, lines, args.device, args.batch_size).detach().cpu().numpy()
        with open(text_features_path, 'wb') as f:
            np.save(f, text_features)
        print(f"Saved text features to {text_features_path}")
    print(f"Text features shape: {text_features.shape}")
    non_spatial_results = attn_results[:, 0] # (N, h, D)
    non_spatial_results_centered = non_spatial_results - non_spatial_results.mean(dim=0, keepdim=True) # (1, h, D)
    # Check how orthogonal the heads are
    print("Checking how orthogonal the heads are...")
    orthogonalities = torch.zeros((non_spatial_results.shape[1], non_spatial_results.shape[1]))
    for batch_idx in range(0, non_spatial_results.shape[0], args.batch_size):
        examples =  non_spatial_results_centered[batch_idx:batch_idx+args.batch_size]
        orthogonalities += torch.abs(torch.einsum('nhd,ngd->nhg', 
                                     F.normalize(examples, dim=-1), 
                                     F.normalize(examples, dim=-1))).sum(dim=0).detach().cpu()
    orthogonalities = orthogonalities.detach().cpu().numpy() / (non_spatial_results.shape[0])
    with open(os.path.join(args.output_dir, f'{name}_{args.model.replace('/', '_')}_orthogonalities.npy'), 'wb') as f:
        np.save(f, orthogonalities)
    plt.figure(figsize=(10, 10))
    plt.imshow(orthogonalities - np.eye(orthogonalities.shape[0]))
    plt.colorbar()
    plt.savefig(os.path.join(args.output_dir, f'{name}_{args.model.replace('/', '_')}_orthogonalities.pdf'))
    plt.close()
    print(f"Saved orthogonalities to {os.path.join(args.output_dir, f'{name}_{args.model.replace('/', '_')}_orthogonalities.npy')}")
    
    if args.compute_text_spans:
        print(f"Non-spatial results shape: {non_spatial_results.shape}")
        print(f"Text features shape: {text_features.shape}")
        for head in range(non_spatial_results.shape[1]):
            reconstruct, results = replace_with_iterative_removal(
                non_spatial_results[:, head].detach().cpu().numpy(),
                text_features,
                lines,
                non_spatial_results.shape[-1],
                non_spatial_results.shape[-1],
                args.device)
            print('--------------------------------')
            print(f"Head {head}")
            for text in results:
                print(text.replace('\n', ''))
            print("--------------------------------")
                                   

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
