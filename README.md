# Radar Semantic Segmentation using CARRADA

Unofficial implementation of the Radar Semantic Segmentation Network (RSS-Net) presented by Kaul et al. in their [paper](https://arxiv.org/abs/2004.03451).

The architecture has been slightly changed to match the [CARRADA dataset](https://arxiv.org/abs/2005.01456).

The CARRADA dataset is available on Arthur Ouaknine's personal web page at this link: [https://arthurouaknine.github.io/codeanddata/carrada](https://arthurouaknine.github.io/codeanddata/carrada).



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


## Running the code

In any case, it is .bold[mandatory] to specify both the path where the CARRADA dataset is located and the path to store the logs and models. Example: I put the Carrada folder in `/datasets`, the path I should specify is `/datasets`. The same way if I store my logs in `/root/logs`. Please run the following command lines while adapting the paths to your settings:
```
$ cd rss-net/rssnet/utils/
$ python set_paths.py --carrada /datasets --logs /root/logs
```


### Training

In order to train a model, a JSON configuration file should be set. An example is proposed in `model_configs/rss_config.json`. Then execute the following command lines for training:
```
$ cd rss-net/rssnet/
$ python train.py --cfg model_configs/rssnet_config.json
```

### Testing

To test a recorded model, you should specify the path to the configuration file recorded in your log folder during training. Per example, if you want to test a model segmenting range-angle representation and you log path has been set to `/root/logs`, you should specify the following path: `/root/logs/carrada/rssnet/range_angle/name_of_the_model/config.json`. This way, execute the following command lines:
```
$ cd rss-net/rssnet/
$ python test.py --cfg /root/logs/carrada/rssnet/range_angle/name_of_the_model/config.json
```


---
### To Do
+ Clear unused options
+ Full Train / Test verifications