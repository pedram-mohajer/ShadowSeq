import os
import csv
import random
import torch
from PIL import Image
from tqdm import tqdm
import pickle

from dataloader import GTSRBSequenceDataset
from dino_attention import DINOAttentionExtractor
from shadow_generator import ShadowGenerator
from shadow_attack import ShadowAttack
from GTSRBClassifier import GTSRBClassifier


def main(
    csv_path="data/gtsrb.csv",
    image_folder="data/gtsrb",
    meta_folder="data/meta_gtsrb",
    pretrained_dino_weights="/.model/dino_gtsrb.pth",
    yolo_weights="models/yolov5/yolov5.pt",
    output_folder="ga_adversarial_shadows",
    GtsrbCnnModel="models/gtsrb_cnn/best_model.pth",
    attn_cache_path="./ShadowGenerator/test_attention_cache.pt",
    shape_shadow="polygon"
):

    
    os.makedirs(output_folder, exist_ok=True)
    results_csv = os.path.join(output_folder, "misclassification_results.csv")
    with open(results_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["sequenceId", "imagename", "class_no_shadow", "class_shadow"])
        writer.writeheader()

    dataset = GTSRBSequenceDataset(csv_path, image_folder)
    print(f"\n Loaded GTSRB with {len(dataset.sequences)} sequences.\n")

    train_set, test_set = dataset.split_train_test(train_count=1106)
    print("Train sequences:", len(train_set))
    print("Test sequences:", len(test_set))
    dataset = test_set  

    dino = DINOAttentionExtractor(
        arch="vit_small",
        patch_size=16,
        pretrained_weights=pretrained_dino_weights,
        meta_dir=meta_folder
    )

    gtsrb_classifier = GTSRBClassifier(
        best_model_path=GtsrbCnnModel,
        img_size=128,
        n_class=43
    )
    shadow_gen = ShadowGenerator()

    if os.path.exists(attn_cache_path):
        print(f"Loading cached attention maps from: {attn_cache_path}")
        test_attention_cache = torch.load(attn_cache_path, weights_only=False)
        #test_attention_cache = torch.load(attn_cache_path)
    else:
        print(f"Attention cache not found. Computing attention maps...")
        test_attention_cache = {}
        for sample in tqdm(dataset, desc="Extracting attention maps for test set"):
            seq_id = sample['sequenceId']
            images = sample['rgb']
            filenames = sample['filenames']
            attn_dict = dino.extract_attention_map_images(images, filenames)
            test_attention_cache[seq_id] = attn_dict
        torch.save(test_attention_cache, attn_cache_path)
        print(f" Attention maps saved to: {attn_cache_path}")

    attacker = ShadowAttack(
        gtsrb_model=gtsrb_classifier,
        dino_model=dino,
        shadow_generator=shadow_gen,
        population_size=20,
        generations=10,
        shape_shadow="polygon"

    )

    for sample in tqdm(dataset, desc="Running GA attacks on sequences"):
        rgb_images = sample['rgb']
        filenames = sample['filenames']
        seq_id = sample['sequenceId']
        real_class = sample['classId']

        print(f"\n🚦 Running attack on Sequence ID {seq_id} (Class {real_class})")

        attn_dict = test_attention_cache[seq_id]

        best_shadowed_imgs, best_params = attacker.attack_sequence(
            rgb_images, filenames, real_class, attn_dict
)
        if(best_shadowed_imgs==None and best_params==None):
            continue

        print(f"✅ Best shadow params for Seq {seq_id}: {best_params}")

        # Save shadowed images
        for img, fname in zip(best_shadowed_imgs, filenames):
            save_path = os.path.join(output_folder, f"shadowed_{seq_id}_{fname}")
            Image.fromarray(img).save(save_path)

        # Evaluate predictions
        preds = gtsrb_classifier.predict_sequence(best_shadowed_imgs, filenames)
        with open(results_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["sequenceId", "imagename", "class_no_shadow", "class_shadow"])
            for fname, pred_class, _ in preds:
                writer.writerow({
                    "sequenceId": seq_id,
                    "imagename": fname,
                    "class_no_shadow": real_class,
                    "class_shadow": pred_class
                })

        print(f"Logged results for Sequence {seq_id}")

if __name__ == "__main__":
    main()
