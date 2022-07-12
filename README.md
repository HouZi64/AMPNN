# AMPNN
The illustration of parallel network with multi-scale fusion and attention mechanism(AMPNN) 

![image](https://user-images.githubusercontent.com/39267242/178009061-10396d1a-c3ee-481c-babe-d1dfc91eb3f3.png)


> **Note**
> More details can be found in paper "Estimating rainfall from surveillance audio based on parallel network with multi-scale fusion and attention mechanism"

# Abstract
Rainfall data have a profound significance for meteorology, climatology, hydrology, and environmental sciences. However, existing rainfall observation methods (including ground-based rain gauges and radar-/satellite-based remote sensing) are not efficient in terms of spatiotemporal resolution and cannot meet the needs of high-resolution application scenarios (urban waterlogging, emergency rescue, etc.). Widespread surveillance cameras have been regarded as alternative rain gauges in existing studies. The surveillance audio, which exploits their nonstop use to record rainfall acoustic signals, should be considered a type of data source to obtain high-resolution and all-weather data. In this study, a method named parallel network based on attention mechanisms and multi-scale fusion (AMPNN) for automatically classifying rainfall levels by surveillance audio is proposed. The proposed model employed a parallel dual-channel network, where the spatial channel extracted the frequency domain correlation while the temporal channel captured the time-domain continuity of the rainfall sound. Additionally, attention mechanisms were used on the two channels to obtain significant spatiotemporal elements. Moreover, a multi-scale fusion method was adopted to fuse different scale features in the SC for more robust performance in complex surveillance scenarios. The experiments showed that our method achieved an accuracy of 84.64% on estimating rainfall levels and outperformed previous research.

# Requirement
* Tensorflow 1.2 
* Keras
* Python 3.6

# Usage
$ python AMPNN_train.py

# Data
Data can be get from 
link：https://pan.baidu.com/s/1JI1XkU8k2DQVQbDdYQ8LhQ 
code：p2vm

# Cite
Please cite our paper if you use this code in your own work:

```bash
@article{
  title={Estimating rainfall from surveillance audio based on parallel network with multi-scale fusion and attention mechanism},
}
```




