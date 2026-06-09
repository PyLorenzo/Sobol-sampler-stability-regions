# Refactoring Summary: Generic Path Support

## Overview

The scripts have been **completely refactored** to remove hardcoded paths and make them easily usable by third parties. All paths are now configurable through command-line arguments and YAML configuration files.

## Key Changes

### 1. **tpm_stability_map.py** - Sobol Sampling Script

#### Before (Hardcoded)
```python
DEFAULT_YAML = "/home/lbaldazzi/Documents/Dottorato/Scripts/yaml_eftcamb/TPM.yaml"
--output /home/lbaldazzi/Documents/.../tpm_stability_map.pkl
```

#### After (Fully Generic)
```bash
# Create config from template
cp config_template.yaml my_config.yaml

# Run with config
python tpm_stability_map.py --config my_config.yaml --base 4096 --outdir ./results

# Override Cobaya YAML path if needed
python tpm_stability_map.py --config my_config.yaml --yaml /another/path.yaml
```

#### New Features
- **`--config`** (required): Path to configuration YAML containing:
  - `cobaya_yaml`: Path to your Cobaya configuration
  - `tpm_ranges`: Parameter ranges to sample
  - `tpm_fixed`: Fixed parameters
  
- **`--yaml`** (optional): Override Cobaya YAML path from config

- **`--outdir`** (optional): Output directory (default: current directory)
  - Output file named: `stability_map.pkl`

#### Configuration File Format (`config_template.yaml`)
```yaml
cobaya_yaml: /path/to/your/config.yaml

tpm_ranges:
  Log_aT: [-7.5, -3.5]
  sig:    [0.4, 3.0]
  M:      [-0.15, 0.015]
  c:      [-0.1, 0.01]

tpm_fixed:
  ombh2: 0.0224
  omch2: 0.118
  tau:   0.055
  logA:  3.05
  ns:    0.965
  H0:    72.0
```

#### Improvements
✓ Dynamic parameter loading from config  
✓ Parameter names no longer hardcoded  
✓ Can be reused for different models/parameters  
✓ Better error handling and validation  
✓ Clearer console output showing loaded configuration  

---

### 2. **Stability_nn.py** - Neural Network Training

#### Before (Hardcoded)
```python
--data "/home/lbaldazzi/Documents/.../tpm_stability_map.pkl"
# Models saved to hardcoded paths
torch.save(model.state_dict(), "/home/lbaldazzi/Documents/.../tpm_stability_model.pt")
```

#### After (Fully Generic)
```bash
# Train with defaults
python Stability_nn.py --data stability_map.pkl --outdir ./models

# With custom parameters
python Stability_nn.py --data stability_map.pkl --outdir ./models \
    --epochs 2000 --batch-size 64 --lr 0.0001 \
    --hidden-dims 128 128 64 32

# With GPU device selection
python Stability_nn.py --data stability_map.pkl --outdir ./models --device cuda:0
```

#### New Command-Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to pickle from `tpm_stability_map.py` |
| `--outdir` | `.` | Output directory for model & scaler |
| `--model-name` | `stability_model.pt` | Name of saved model file |
| `--scaler-name` | `stability_scaler.pkl` | Name of saved scaler file |
| `--epochs` | 1000 | Number of training epochs |
| `--batch-size` | 32 | Batch size for training |
| `--lr` | 0.001 | Learning rate |
| `--val-split` | 0.2 | Validation split (20%) |
| `--hidden-dims` | [64, 64, 32] | Hidden layer dimensions |
| `--device` | auto-detect | Device (cuda, cuda:0, cpu, mps) |
| `--early-stopping` | 50 | Patience for early stopping |
| `--seed` | 42 | Random seed |

#### Improvements
✓ Fully configurable network architecture  
✓ Flexible device management (auto-detect GPU/CPU/MPS)  
✓ Auto-detecting best model during training  
✓ Early stopping with configurable patience  
✓ Batch training with data loaders  
✓ Better inference instructions in output  
✓ Informative console output with parameter counts  

---

### 3. **config_template.yaml** - New Configuration Template

A ready-to-use template file for easy customization:

```yaml
cobaya_yaml: /path/to/your/config.yaml

tpm_ranges:
  Log_aT: [-7.5, -3.5]
  sig:    [0.4, 3.0]
  M:      [-0.15, 0.015]
  c:      [-0.1, 0.01]

tpm_fixed:
  ombh2: 0.0224
  omch2: 0.118
  tau:   0.055
  logA:  3.05
  ns:    0.965
  H0:    72.0

# Optional: training config
nn_config:
  hidden_dims: [64, 64, 32]
  epochs: 1000
  learning_rate: 0.001
```

---

## Usage Workflow

### For New Users / Different Models

```bash
# 1. Clone and install
git clone <repo>
cd Sobol-sampler-stability-regions-TPM
pip install -r requirements.txt

# 2. Prepare your Cobaya configuration
# (assuming you have: my_cobaya_config.yaml)

# 3. Create a project config
cp config_template.yaml my_project.yaml
# Edit my_project.yaml with your paths and parameters

# 4. Run sampling
python tpm_stability_map.py --config my_project.yaml --base 4096 --outdir ./results

# 5. Train neural network
python Stability_nn.py --data ./results/stability_map.pkl --outdir ./models

# 6. Visualize results
python tpm_stability_viz.py ./results/stability_map.pkl --outdir ./figures
```

---

## Backward Compatibility Notes

- Old hardcoded scripts are **not** compatible with new versions
- If you have old pickles, they still work; just point to them with `--data`
- Output pickle format is **unchanged**
- All changes are in I/O and configuration, not in algorithms

---

## Benefits of Refactoring

✅ **Reusable**: Use the same code for different models/parameters  
✅ **Shareable**: No hardcoded user paths; anyone can run it  
✅ **Maintainable**: Configuration separate from code  
✅ **Scalable**: Easy to add new parameters or extend functionality  
✅ **Professional**: Follows software engineering best practices  
✅ **Documented**: Clear usage examples and help messages  

---

## Example: Using Different Parameters

Suppose you want to sample a 3-parameter space instead:

```yaml
# my_3param_config.yaml
cobaya_yaml: /my/cobaya/config.yaml

tpm_ranges:
  alpha: [0.1, 1.0]
  beta:  [0.5, 2.0]
  gamma: [-1.0, 1.0]

tpm_fixed:
  ombh2: 0.0224
  omch2: 0.118
  H0:    72.0
```

Then just run:
```bash
python tpm_stability_map.py --config my_3param_config.yaml
```

No code changes needed!

---

## Troubleshooting

### Error: "Configuration file not found"
→ Check that `--config` points to an existing YAML file

### Error: "cobaya_yaml not found"
→ Check that the path in your config YAML is correct and accessible

### Error: "Missing required configuration key"
→ Ensure your config has `cobaya_yaml`, `tpm_ranges`, and `tpm_fixed`

### Model training too slow?
→ Try `--device cuda:0` if you have NVIDIA GPU  
→ Reduce `--batch-size` or `--hidden-dims` if out of memory

---

## Questions?

For issues or questions, please open an issue on GitHub!
