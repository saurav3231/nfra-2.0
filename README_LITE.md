# NFRA Lite — Documentation

**Optimized for Legacy & Low-Power Hardware**

Created by Saurav Bhandari

---

## Overview

**NFRA Lite** is the most lightweight and efficient variant of the NFRA system. It is specifically designed to run on very old and low-power CPUs, such as the **Intel Core i5-337U** (2012).

### Key Features

- Extremely small model size (60M–120M parameters)
- Very high sparsity (92–97%)
- Optimized for CPU inference
- Low memory footprint
- Aggressive energy management

---

## Quick Start

```python
from nfra.models import create_nfra_lite

# Create the model
model = create_nfra_lite()

# Run inference
import torch
input_ids = torch.randint(0, 50257, (1, 128))

with torch.no_grad():
    outputs = model(input_ids, energy_budget=0.5)
```

---

## Hardware Requirements

| Hardware                  | Expected Speed     | Memory Usage | Recommended |
|---------------------------|--------------------|--------------|-------------|
| Intel Core i5-337U (2012) | 5–10+ tokens/sec   | ~400–550 MB  | Yes         |
| Raspberry Pi 4/5          | 8–14 tokens/sec    | ~400 MB      | Yes         |
| Modern Laptop CPU         | 25–40+ tokens/sec  | ~400 MB      | Yes         |

---

## Configuration

NFRA Lite uses the following aggressive settings:

- Hidden Size: 384
- Number of Layers: 8
- Fractal Scales: [1, 2]
- Energy Budget: 0.5 (very aggressive)
- All advanced features disabled

---

## When to Use NFRA Lite

Use **NFRA Lite** when:
- You have old hardware (pre-2015 CPUs)
- You need maximum inference speed
- Memory is limited (< 1GB available)
- You want the simplest possible model

Use **NFRA Mid** or **NFRA Max** when you have more powerful hardware and want better intelligence.

---

## Performance Tips

1. Always use `energy_budget=0.5` or lower for best speed
2. Keep sequence length ≤ 256 for best performance on old CPUs
3. Use batch size 1 or 2 on very weak hardware

---

*NFRA Lite is part of the NFRA 2.0 project by Saurav Bhandari.*