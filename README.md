This project develops an end-to-end pipeline for automated calorie estimation from foodsegdataset images by integrating semantic segmentation 
with a curated nutrition database. Two model variants are implemented and compared on the FoodSeg103 dataset: a U-Net baseline at 256×256 resolution, 
and a transformer-based SegFormer-B3 at 384×512 resolution with polynomial learning-rate decay and exponential moving-average (EMA) stabilisation. 
The U-Net achieves a validation pixel accuracy of 57.3%, while SegFormer-B3 improves to 84.4% pixel accuracy and 44.4% mean IoU, 
demonstrating significantly better class separation and boundary detection. Predicted masks are converted into calorie estimates by mapping segmented classes to energy densities (kcal/g) 
derived from USDA FoodData Central and normalising portion weights to a 200 g reference serving. 
The system outputs per-item and total calorie values with visual overlays. 
Limitations include the absence of physical scale, challenges with thin or mixed foods, and taxonomy mismatches between dataset labels and nutrition entries.
Nonetheless, results show that transformer-based segmentation provides a robust foundation for downstream calorie estimation, and future work can integrate scale cues, enhanced loss functions, 
and richer nutrition mappings to improve accuracy and usability.
Overview
This project implements an end-to-end pipeline for calorie estimation from a foodsegdataset images.
It combines semantic segmentation (U-Net & SegFormer-B3) with a USDA FoodData Central kcal/g map to produce per-item and total calorie estimates.


Code  Structure

calorie_estimation.py → Early data loading and plots,and top 10 class like that.

project_code_final__2.py → Main training pipeline on unet model.

segmodel_2.py → SegFormer-B3 model definition & loading, training pipeline segmodel.

results.py →Show the output of my best model which got miou 0.44 ,upload a image and show the calories estimation
