# UNET1_2D Models Directory

## Purpose
Storage location for trained UNET1_2D models and their hyperparameters.

## Structure
Each model training run creates a subdirectory:
```
unet1_2d_models/
├── UNET2D_4ch_bs16_ep100_lr0.001/
│   ├── model.keras           (trained weights)
│   └── hyperparameters.json  (config + timestamp)
├── UNET2D_4ch_bs16_ep100_lr0.001_v2/
│   ├── model.keras
│   └── hyperparameters.json
└── ...
```

## Naming Convention
`UNET2D_4ch_bs{batch}_ep{epochs}_lr{learning_rate}/`
- `4ch`: 4-channel input (log_counts, mean_theta_sq_z, var_poca_z, std_theta)
- `bs`: batch size
- `ep`: number of epochs
- `lr`: learning rate

## Usage
UNET1_2D.py automatically saves models here using the hyperparameters naming scheme.

## Loading Trained Models
```python
from tensorflow import keras
model = keras.models.load_model('./unet1_2d_models/UNET2D_4ch_bs16_ep100_lr0.001/model.keras')
```
