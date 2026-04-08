# [TMM 2021] Discover Micro-influencers for Brands via Better Understanding

## News
A revised version of the paper has been published.

## Abstract

<div align=center>
<img src="https://github.com/Mysteriousplayer/TMM21-CAMERA/blob/main/fig1.png" width="50%">
</div>

> With the rapid development of the influencer marketing industry in recent years, the cooperation between brands and micro-influencers on marketing has achieved much attention. As a key sub-task of influencer marketing, micro-influencer recommendation is gaining momentum. However, in influencer marketing campaigns, it is not enough to only consider marketing effectiveness. Towards this end, we propose a concept-based micro-influencer ranking framework, to address the problems of marketing effectiveness and self-development needs for the task of micro-influencer recommendation. Marketing effectiveness is improved by concept-based social media account representation and a micro-influencer ranking function. We conduct social media account representation from the perspective of historical activities and marketing direction. And two adaptive learned metrics, endorsement effect score and micro-influencer influence score, are defined to learn the micro-influencer ranking function. To meet self-development needs, we design a bi-directional concept attention mechanism to focus on brands’ and micro-influencers’ marketing direction over social media concepts. Interpretable concept-based parameters are utilized to help brands and micro-influencers make marketing decisions.Extensive experiments conducted on a real-world dataset demonstrate the advantage of our proposed method compared with the state-of-the-art methods.

## Framework

<div align=center>
<img src="https://github.com/Mysteriousplayer/TMM21-CAMERA/blob/main/fig2.png">
</div>

<div align=center>
<img src="https://github.com/Mysteriousplayer/TMM21-CAMERA/blob/main/fig3.png" width="50%">
</div>

-We propose the CAMERA to address marketing effectiveness and self-development needs together in the micro-influencer recommendation task, which successfully understands the marketing intent of social media accounts at a fine-grained level. 
-We design the COSMIC and the BCAM to learn social media account representation from the perspective of historical activities and marketing direction. Meanwhile, interpretable concept-based parameters from the two perspectives can be utilized to help brands and micro-influencers make decisions. 
-We model endorsement information and micro-influencer influence information in micro-influencer ranking, where two novel adaptive learned metrics (endorsement effect score and micro-influencer influence score) are defined to learn a better micro-influencer ranking function. 
-Recommendation performance analysis and recommendation interpretability analysis demonstrate the advantage of our proposed method compared with the state-of-the-art methods. Especially, we implement a new application scenario of keywords-based micro-influencer search based on our proposed method and demonstrate the effectiveness of it.

## Installation
Install all requirements required to run the code on a Python 3.6.13 by:
> First, you need activate a new conda environment.
> 
> pip install -r requirements.txt

## Datasets
Brand-micro-influencer dataset: plase contact email: gantian@sdu.edu.cn.

## Training

```
python camera_v2.py
```

## Cite
If you found our work useful for your research, please cite our work:
```
@ARTICLE{wang_2022, 
author={Wang, Shaokun and Gan, Tian and Liu, Yuan and Zhang, Li and Wu, JianLong and Nie, Liqiang}, 
journal={IEEE Transactions on Multimedia}, 
title={Discover Micro-Influencers for Brands via Better Understanding}, 
year={2022}, 
volume={24}, 
pages={2595-2605}}
```
