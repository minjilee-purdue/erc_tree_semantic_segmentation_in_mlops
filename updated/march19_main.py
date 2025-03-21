# main.py


import os
import torch
import argparse
import time
import datetime
import logging

from torchvision import transforms
from segment_anything import sam_model_registry

# Import your custom modules
from march19_dataset import CedarTreeDataset
from march19_sam_trainer import SamFineTuner
from march19_sam_finetuning_utility import (
    load_sam_model,
    prepare_data_loaders,
    test_model_and_evaluate,
    run_model_comparison,
    setup_experiment
)

def parse_args():
    parser = argparse.ArgumentParser(description="SAM Fine-tuning for Cedar Tree Segmentation")
    
    # Data paths
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing masks")
    parser.add_argument("--bbox_dir", type=str, required=True, help="Directory containing bounding boxes")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], 
                        help="SAM model type to use")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to SAM model checkpoint")
    parser.add_argument("--compare_models", action="store_true", help="Compare multiple model architectures")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Experiment settings
    parser.add_argument("--experiment_name", type=str, default="cedar_sam_finetune", 
                        help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():

    args = parse_args()

    # Setup logging
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "experiment.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    print(f"Using device: {device}")
    
    # Set up experiment
    if args.compare_models:
        model_configs = [
            ('vit_b', args.checkpoint_path.replace("vit_h", "vit_b").replace("vit_l", "vit_b")),
            ('vit_l', args.checkpoint_path.replace("vit_h", "vit_l").replace("vit_b", "vit_l")),
            ('vit_h', args.checkpoint_path.replace("vit_b", "vit_h").replace("vit_l", "vit_h"))
        ]
    else:
        model_configs = [(args.model_type, args.checkpoint_path)]
    
    experiment_config = setup_experiment(
        args.experiment_name, 
        model_configs, 
        args.image_dir, 
        args.mask_dir, 
        args.bbox_dir
    )
    
    # Create dataset
    print("Creating dataset...")
    dataset = CedarTreeDataset(
        image_dir=args.image_dir, 
        mask_dir=args.mask_dir, 
        bbox_dir=args.bbox_dir, 
        transform=None
    )
    
    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader = prepare_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    if args.compare_models:
        # Run model comparison
        print("Running model comparison...")
        results = run_model_comparison(
            dataset=dataset,
            model_configs=model_configs,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs
        )
    else:
        # Train single model
        print(f"Training single model: {args.model_type}")
        
        # Load model
        sam = load_sam_model(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint_path,
            device=device
        )
        
        # Initialize fine-tuner
        fine_tuner = SamFineTuner(
            sam_model=sam,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Train model
        print("Starting training...")
        start_time = time.time()
        history = fine_tuner.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_dir=experiment_config["directories"]["checkpoints_dir"]
        )
        training_time = time.time() - start_time
        print(f"Training completed in {datetime.timedelta(seconds=int(training_time))}")
        
        # Test model
        print("Testing model...")
        test_metrics = test_model_and_evaluate(
            model=sam,
            dataset=dataset,
            device=device,
            num_samples=30,
            save_dir=experiment_config["directories"]["results_dir"]
        )
        
        # Print test metrics
        print("\nTest Results:")
        print(f"Average Dice Score: {test_metrics['avg_dice']:.4f}")
        print(f"Average IoU: {test_metrics['avg_iou']:.4f}")
        print(f"Average Precision: {test_metrics['avg_precision']:.4f}")
        print(f"Average Recall: {test_metrics['avg_recall']:.4f}")
        print(f"Average Inference Time: {test_metrics['avg_inference_time']:.4f} seconds")
    
    print("\nExperiment completed!")
    print(f"Results saved to: {experiment_config['directories']['experiment_dir']}")

if __name__ == "__main__":
    main()




'''
(erc) minjilee@pop-os:~$ python /home/minjilee/prj/march19_main.py
usage: march19_main.py [-h] --image_dir IMAGE_DIR --mask_dir MASK_DIR --bbox_dir BBOX_DIR
                       [--model_type {vit_b,vit_l,vit_h}] --checkpoint_path CHECKPOINT_PATH
                       [--compare_models] [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                       [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                       [--val_split VAL_SPLIT] [--num_workers NUM_WORKERS]
                       [--experiment_name EXPERIMENT_NAME] [--seed SEED]
march19_main.py: error: the following arguments are required: --image_dir, --mask_dir, --bbox_dir, --checkpoint_path
(erc) minjilee@pop-os:~$ python /home/minjilee/prj/march19_main.py \
    --image_dir /home/minjilee/Desktop/dataset/images \
    --mask_dir /home/minjilee/Desktop/dataset/masks_grey \
    --bbox_dir /home/minjilee/Desktop/dataset/bbox_txt \
    --checkpoint_path /home/minjilee/Downloads/sam_vit_b_01ec64.pth \
    --model_type vit_b
Using device: cuda
Experiment 'cedar_sam_finetune' set up in directory: experiments/cedar_sam_finetune_20250319_165248
Configuration saved to: experiments/cedar_sam_finetune_20250319_165248/experiment_config.json
Creating dataset...

Found 106 images
Found 106 masks
Found 106 bbox files
Found 106 valid matching samples

Visualizing first sample:
  Image: MAX_0009_enhanced_final.jpg
  Mask: MAX_0009_enhanced_final.png
  Bbox: MAX_0009_enhanced_final_bboxes.txt
  Bounding box: [array([4588, 2087, 5122, 2565])]
  Visualization saved to first_sample_visualization.png
Preparing data loaders...
Training dataset size: 85
Validation dataset size: 21
Training single model: vit_b
Loading vit_b model from /home/minjilee/Downloads/sam_vit_b_01ec64.pth...
/home/minjilee/prj/march19_sam_finetuning_utility.py:28: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model_state = torch.load(checkpoint_path, map_location=device)
Model loaded successfully and moved to cuda
/home/minjilee/miniconda3/envs/erc/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
Starting training...
Epoch 1 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.52it/s, loss=0.0404, dice=0.9244]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.87it/s]

Epoch 1/10:
Train Loss: 0.1102, Train Dice: 0.8159
Val Loss: 0.0701, Val Dice: 0.8696
Saved best model checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_best.pth
Epoch 2 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.75it/s, loss=0.0839, dice=0.8735]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.95it/s]

Epoch 2/10:
Train Loss: 0.0899, Train Dice: 0.8425
Val Loss: 0.0678, Val Dice: 0.8729
Saved best model checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_best.pth
Epoch 3 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.76it/s, loss=0.0739, dice=0.8670]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.89it/s]

Epoch 3/10:
Train Loss: 0.0817, Train Dice: 0.8564
Val Loss: 0.0678, Val Dice: 0.8722
Epoch 4 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.74it/s, loss=0.0796, dice=0.8557]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.83it/s]

Epoch 4/10:
Train Loss: 0.0820, Train Dice: 0.8565
Val Loss: 0.0678, Val Dice: 0.8728
Saved best model checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_best.pth
Epoch 5 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.73it/s, loss=0.0491, dice=0.9027]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.85it/s]

Epoch 5/10:
Train Loss: 0.0803, Train Dice: 0.8584
Val Loss: 0.0677, Val Dice: 0.8717
Saved best model checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_best.pth
Saved checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_epoch_5.pth
Epoch 6 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.72it/s, loss=0.0766, dice=0.8485]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.84it/s]

Epoch 6/10:
Train Loss: 0.0762, Train Dice: 0.8652
Val Loss: 0.0658, Val Dice: 0.8744
Saved best model checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_best.pth
Epoch 7 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.72it/s, loss=0.0882, dice=0.8197]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.84it/s]

Epoch 7/10:
Train Loss: 0.0724, Train Dice: 0.8723
Val Loss: 0.0659, Val Dice: 0.8754
Epoch 8 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.72it/s, loss=0.0987, dice=0.8039]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.81it/s]

Epoch 8/10:
Train Loss: 0.0762, Train Dice: 0.8668
Val Loss: 0.0667, Val Dice: 0.8751
Epoch 9 Training: 100%|███████████████| 43/43 [00:07<00:00,  5.69it/s, loss=0.0588, dice=0.9018]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.82it/s]

Epoch 9/10:
Train Loss: 0.0726, Train Dice: 0.8730
Val Loss: 0.0650, Val Dice: 0.8778
Saved best model checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_best.pth
Epoch 10 Training: 100%|██████████████| 43/43 [00:07<00:00,  5.71it/s, loss=0.0512, dice=0.9095]
Validation: 100%|███████████████████████████████████████████████| 11/11 [00:02<00:00,  4.82it/s]

Epoch 10/10:
Train Loss: 0.0726, Train Dice: 0.8730
Val Loss: 0.0659, Val Dice: 0.8761
Saved checkpoint to experiments/cedar_sam_finetune_20250319_165248/checkpoints/sam_finetuned_epoch_10.pth
Training completed in 0:01:40
Testing model...
Processed test sample 61, Dice: 0.9509, IoU: 0.9063, Precision: 0.9580, Recall: 0.9438
Processed test sample 11, Dice: 0.9118, IoU: 0.8379, Precision: 0.9401, Recall: 0.8851
Processed test sample 98, Dice: 0.8947, IoU: 0.8095, Precision: 0.9012, Recall: 0.8884
Processed test sample 84, Dice: 0.8287, IoU: 0.7075, Precision: 0.8875, Recall: 0.7773
Processed test sample 7, Dice: 0.9619, IoU: 0.9266, Precision: 0.9637, Recall: 0.9602
Processed test sample 35, Dice: 0.9234, IoU: 0.8578, Precision: 0.9432, Recall: 0.9045
Processed test sample 57, Dice: 0.8469, IoU: 0.7345, Precision: 0.9665, Recall: 0.7537
Processed test sample 14, Dice: 0.8832, IoU: 0.7909, Precision: 0.8096, Recall: 0.9717
Processed test sample 67, Dice: 0.8790, IoU: 0.7841, Precision: 0.8827, Recall: 0.8753
Processed test sample 25, Dice: 0.8350, IoU: 0.7168, Precision: 0.8904, Recall: 0.7862
Processed test sample 4, Dice: 0.8419, IoU: 0.7270, Precision: 0.8103, Recall: 0.8761
Processed test sample 65, Dice: 0.8741, IoU: 0.7764, Precision: 0.8124, Recall: 0.9460
Processed test sample 82, Dice: 0.8685, IoU: 0.7675, Precision: 0.8612, Recall: 0.8758
Processed test sample 26, Dice: 0.8783, IoU: 0.7831, Precision: 0.8438, Recall: 0.9159
Processed test sample 66, Dice: 0.8864, IoU: 0.7959, Precision: 0.8663, Recall: 0.9074
Processed test sample 13, Dice: 0.8438, IoU: 0.7298, Precision: 0.8862, Recall: 0.8053
Processed test sample 28, Dice: 0.9041, IoU: 0.8250, Precision: 0.9233, Recall: 0.8857
Processed test sample 24, Dice: 0.9145, IoU: 0.8425, Precision: 0.8931, Recall: 0.9370
Processed test sample 56, Dice: 0.9555, IoU: 0.9148, Precision: 0.9518, Recall: 0.9593
Processed test sample 94, Dice: 0.9045, IoU: 0.8256, Precision: 0.8681, Recall: 0.9440
Processed test sample 55, Dice: 0.9510, IoU: 0.9066, Precision: 0.9655, Recall: 0.9370
Processed test sample 60, Dice: 0.9472, IoU: 0.8998, Precision: 0.9608, Recall: 0.9341
Processed test sample 12, Dice: 0.8145, IoU: 0.6871, Precision: 0.7693, Recall: 0.8654
Processed test sample 68, Dice: 0.8894, IoU: 0.8008, Precision: 0.8348, Recall: 0.9516
Processed test sample 46, Dice: 0.9517, IoU: 0.9079, Precision: 0.9602, Recall: 0.9434
Processed test sample 2, Dice: 0.9182, IoU: 0.8488, Precision: 0.8964, Recall: 0.9412
Processed test sample 88, Dice: 0.8631, IoU: 0.7592, Precision: 0.7979, Recall: 0.9400
Processed test sample 102, Dice: 0.9001, IoU: 0.8184, Precision: 0.8587, Recall: 0.9457
Processed test sample 70, Dice: 0.9101, IoU: 0.8350, Precision: 0.8760, Recall: 0.9468
Processed test sample 91, Dice: 0.9073, IoU: 0.8304, Precision: 0.8671, Recall: 0.9515
Test metrics saved to experiments/cedar_sam_finetune_20250319_165248/results/test_metrics.json
Metrics distribution visualization saved to experiments/cedar_sam_finetune_20250319_165248/results/metrics_distribution.png

Test Results:
Average Dice Score: 0.8947
Average IoU: 0.8118
Average Precision: 0.8882
Average Recall: 0.9052
Average Inference Time: 0.0771 seconds

Experiment completed!
Results saved to: experiments/cedar_sam_finetune_20250319_165248

'''