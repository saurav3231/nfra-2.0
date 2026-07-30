# NFRA Lite - Installation & Usage Guide

**For Intel Core i5-337U and Similar Old Hardware**

Created by Saurav Bhandari

---

## Prerequisites

Before starting, make sure you have:

1. **Python 3.9 or 3.10** installed
2. **PyTorch** already installed (`pip show torch`)
3. Windows 10/11 (or compatible OS)

---

## Method 1: Simple Installation (Recommended)

### Step 1: Download the Project

1. Download the entire `NFRA-2.0` folder
2. Extract it to a location of your choice (e.g., `C:\NFRA-2.0`)

### Step 2: Install Required Packages

Open **Command Prompt** and run:

```cmd
cd C:\NFRA-2.0
pip install numpy pyyaml tqdm
```

> **Note**: Do **not** install PyTorch again if you already have it.

### Step 3: Test NFRA Lite

Run this command:

```cmd
python -c "from src.nfra.models import create_nfra_lite; model = create_nfra_lite(); print('NFRA Lite loaded successfully!')"
```

If you see **"NFRA Lite loaded successfully!"**, everything is working.

### Step 4: Run NFRA Lite

You can now use it like this:

```python
from src.nfra.models import create_nfra_lite
import torch

model = create_nfra_lite()
model.eval()

# Example inference
input_ids = torch.randint(0, 50257, (1, 64))
with torch.no_grad():
    outputs = model(input_ids, energy_budget=0.5)
    
print("Inference successful!")
print(f"Output shape: {outputs['logits'].shape}")
```

---

## Method 2: Create .exe File (Optional)

If you want a standalone executable:

### Step 1: Install PyInstaller

```cmd
pip install pyinstaller
```

### Step 2: Build the Executable

```cmd
cd C:\NFRA-2.0
python build_nfra_lite_exe.py
```

### Step 3: Run the .exe

After building, you will find the executable here:

```
C:\NFRA-2.0\dist\NFRA_Lite.exe
```

You can copy this `.exe` anywhere and run it (as long as PyTorch is installed on the system).

---

## Performance Tips for i5-337U

1. Always use `energy_budget=0.5` or lower
2. Keep sequence length ≤ 128–256 for best speed
3. Use batch size = 1 on very old hardware
4. Close other applications while running

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'nfra'"
→ Make sure you are running from the `NFRA-2.0` folder.

### Slow performance
→ Use `energy_budget=0.4` or lower.

### Out of memory
→ Reduce `max_seq_length` in the config.

---

## Support

This project was created by **Saurav Bhandari**.

For issues, refer to the main README or contact the creator.

---

**Good luck testing NFRA Lite on your i5-337U!**