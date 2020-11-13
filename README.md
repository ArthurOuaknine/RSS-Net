# Radar Semantic Segmentation using CARRADA

Unofficial implementation of the Radar Semantic Segmentation Network (RSS-Net) presented by Kaul et al. in their [paper](https://arxiv.org/abs/2004.03451).

The architecture has been slightly changed to match the [CARRADA dataset](https://arxiv.org/abs/2005.01456).

The CARRADA dataset is available on Arthur Ouaknine's personal web page at this link: [https://arthurouaknine.github.io/codeanddata/carrada](https://arthurouaknine.github.io/codeanddata/carrada).

---

## Installation

1. Clone the repo:
```
$ git clone https://github.com/ArthurOuaknine/RSS-Net.git
```

2. Install this repository and the dependencies using pip and conda:
```
$ cd /path/to/rss-net/repo
$ pip install -e
$ pip install -r requirements.txt
```

3. Download the CARRADA Dataset [here](https://arthurouaknine.github.io/codeanddata/carrada)

4. Optional. To uninstall this package, run:
```
$ pip uninstall rss-net
```
---

---
### To Do
+ File: requirements.txt
+ Script: set_path.py (config.ini)
+ Script: maint_test.py
+ Clear unused options
+ Comment code
+ README (install package, command line for train, command line for test)
+ Full Train / Test verifications