# Probabilistic 3D Multi-Object Tracking for Autonomous Driving

Hsu-kuang Chiu<sup>1</sup>, Antonio Prioletti<sup>2</sup>, Jie Li<sup>2</sup>, Jeannette Bohg<sup>1</sup>

<sup>1</sup>Stanford University, <sup>2</sup>Toyota Research Institute

First Place Award, NuScenes Tracking Challenge, at AI Driving Olympics Workshop, NeurIPS 2019.

 

## Abstract

We present our on-line tracking method, which wins the first place award of the [NuScenes Tracking Challenge](https://www.nuscenes.org/tracking)[1], held in the AI Driving Olympics Workshop at NeurIPS 2019. Our technical report is available in [arxiv](https://arxiv.org/abs/2001.05673). Our code will be available soon.

<img align="center" src="images/architecture.jpg">

## Quantitative Results

The following table shows our quantitative tracking results for the validation set of NuScenes: evaluation in terms of overall Average Multi-Object Tracking Accuracy (AMOTA) and individual AMOTA for each object category in comparison with the [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)[2] baseline method.

Method | Overall | bicycle | bus | car | motorcycle | pedestrian | trailer | truck
---------- | --: | --: | --: | --: | --: | --: | --: | --: 
AB3DMOT[2] | 17.9 |  0.9 | 48.9 | 36.0 |  5.1 |  9.1 | 11.1 | 14.2
Our method | 56.1 | 27.2 | 74.1 | 73.5 | 50.6 | 75.5 | 33.7 | 58.0

We can see that our method improves the AMOTAs significantly, especially for the smaller objects, such as pedestrians.

The NuScenes Tracking Challenge organizer shared the test set performance of the top 3 participants and the AB3DMOT [2] baseline, as shown in the following table. The full tracking challenge leaderboard will be released to public soon by the organizer.

Rank       | Team Name               | Overall
:--------: | :--                     | --:
1          | StanfordIPRL-TRI (Ours) | 55.0
2          | VV_team                 | 37.1
3          | CenterTrack             | 10.8
baseline   | AB3DMOT [2]             | 15.1


## Qualitative Results

The following figures are the bird-eye-view visualization of the tracking results from the AB3DMOT[2] baseline and our method. For this scene, we draw all the car bounding boxes from all frames of the same scene in a single plot. Different colors represent different tracking ids. We also show the ground-truth annotations and the input detections as the references.


AB3DMOT[2] | Ours 
:---:|:---:
<img src="images/ab3dmot.jpg" width=400px height=400px> | <img src="images/ours.jpg" width=400px height=400px>


Input Detections | Ground-Truth
:---:|:---:
<img src="images/detection.jpg" width=400px height=400px> | <img src="images/gt.jpg" width=400px height=400px>

We can see that our method is able to better track the object making a sharp turn.


## Acknowledgement
- We implemented our method on top of [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)[2]'s open-source code.

- Toyota Research Institute ("TRI") provided funds to assist the authors with their research but this article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota entity.


## References
- \[1\] *"nuScenes: A multimodal dataset for autonomous driving"*, Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom, arXiv:1903.11027, 2019.
- \[2\] *"A Baseline for 3D Multi-Object Tracking"*, Xinshuo Weng and Kris Kitani, arXiv:1907.03961, 2019.


