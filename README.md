# ORCA

Please modify `DATA_DIR` to the desired destination folder to save EuroSAT.

To reproduce the results of `ORCA-B` on the `EuroSAT` dataset using the `ViT-B/32` backbone with 10 concepts, please run:

    python main.py --model ViT-B/32 --n-concepts 10 --method base

To reproduce the results of `ORCA-R` on the `EuroSAT` dataset using the `ViT-B/32` backbone with 10 concepts, please run:

    python main.py --model ViT-B/32 --n-concepts 10 --method rank