# 🚀 EdgeTAM-SAMURAI: On-Device Tracking with Motion-Aware Memory.


> **Fast on-device tracking that remembers everything**  
> Combining EdgeTAM's speed with SAMURAI's memory bank for next-level video object tracking. 

<div align="center">

https://github.com/user-attachments/assets/0ab449db-d42c-4c04-8c34-a25d2a33ddd2

</div>

---

## ✨ Features

- 🚀 22× faster than SAM 2 (inherited from EdgeTAM)
- 🧠 Smart memory for long-term tracking
- 📱 16 FPS on iPhone 15 Pro Max
- 🎯 Handles occlusions like a pro

## 📖 Overview
| Component | What It Brings |
|-----------|---------------|
| **EdgeTAM** | Mobile-optimized architecture, 22× speed boost, on-device efficiency |
| **SAMURAI** | Motion-aware memory bank, long-term tracking, occlusion handling |
| **Our Implementation** | Seamless integration of speed + memory for tracking |



## 🛠️ Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/m-dear/EdgeTAM-Samurai.git
cd EdgeTAM-Samurai
# Install dependencies
pip install -e .
```

### 🎬 Real-time implementation

```python
import torch
from edge_sam2.build_sam import build_sam2_object_tracker

# Load the model
checkpoint = "checkpoints/edgetam.pt"
model_cfg = "edgetam.yaml"
predictor = build_sam2_object_tracker(
      config_file=model_cfg,
      ckpt_path=checkpoint,
      device="cuda:0",  # Use "cpu" for CPU inference
      # This implementation is for a single object
      num_objects=1, 
      verbose=False
  )
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:        
        # Track object
        if is_first_frame:
            # Provide the initial bounding box to track a new object
            sam_out = self.predictor.track_new_object(img=img_rgb, box=bbox)
            is_first_frame = False
        else:
            # Track all previously identified objects in subsequent frames
            sam_out = self.predictor.track_all_objects(img=img_rgb)
```
⚠️ Implementation Note
Currently, only the `build_sam2_object_tracker` function implements the EdgeTAM and SAMURAI integration. Other functions may use standard SAM2 functionality.

### Demo code
```bash
cd EdgeTAM-Samurai
# Run the demo script
python demo/run_realtime.py
```
Use mouse to draw bounding box on the video frame to track a new object, or press `space or enter` to track all previously identified objects.


## 🙏 Acknowledgments
This project builds upon the work of the EdgeTAM authors at Meta Reality Labs and the SAMURAI authors. We extend their efforts by integrating EdgeTAM's speed with SAMURAI's memory bank to create a powerful on-device tracking solution.


## 📄 Citation
```bibtex
@article{zhou2025edgetam,
  title={EdgeTAM: On-Device Track Anything Model},
  author={Zhou, Chong and Zhu, Chenchen and Xiong, Yunyang and Suri, Saksham and Xiao, Fanyi and Wu, Lemeng and Krishnamoorthi, Raghuraman and Dai, Bo and Loy, Chen Change and Chandra, Vikas and Soran, Bilge},
  journal={arXiv preprint arXiv:2501.07256},
  year={2025}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
@misc{yang2024samurai,
  title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
  author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
  year={2024},
  eprint={2411.11922},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.11922}, 
}
```
## 🤝 Contributors
- **EdgeTAM** (Meta Reality Labs) for efficient on-device tracking
- **SAMURAI** for motion-aware memory mechanisms  
- **SAM 2** (Meta) for the foundational segmentation model
- **[zdata-inc](https://github.com/zdata-inc/sam2_realtime)** for real-time implementation insights

---

<div align="center">

[🌟 Star us on GitHub](https://github.com/m-dear/EdgeTAM-Samurai.git) 

</div>
