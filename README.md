<div align="center">
<h1> Awesome-Scene-Graph-for-VL-Learning </h1> 
</div>




# 🎨 Introduction 
A scene graph is a topological structure representing a scene described in text, image, video, or etc. 
In this graph, the nodes correspond to object bounding boxes with their category labels and attributes, while the edges represent the pair-wise relationships between objects.
  <p align="center">
  <img src="assets/intro1.png" width="75%">
</p>



# 📕 Table of Contents
- [🎨 Introduction](#-introduction)
- [📕 Table of Contents](#-contents)
- [🌷 Scene Graph Datasets](#-datasets)
- [🍕 Scene Graph Generation](#-scene-graph-generation)
  - [Image Scene Graph Generation](#)
  - [Video Scene Graph Generation](#)
  - [Audio Scene Graph Generation](#)
- [🥝 Scene Graph Application](#-scene-graph-application)
  - [Image Retrieval](#)
  - [Visual Question Answering](#)
  - [Image Captioning](#)
  - [Video Retrieval](#)
- [⭐️ Star History](#️-star-history)

# 🌷 Scene Graph Datasets
<p align="center">

| Dataset |  Modality  |   Obj. Class  | BBox | Rela. Class | Triplets | Instances | 
|:--------:|:--------:|:--------:| :--------:|  :--------:|  :--------:|  :--------:|
| [Visual Phrase](https://vision.cs.uiuc.edu/phrasal/) | Image | 8 | 3,271 | 9 | 1,796 | 2,769 |
| [Scene Graph](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf) | Image | 266 | 69,009 | 68 | 109,535 | 5,000 |
| [VRD]()  | Image | 100 | - | 70 | 37,993 | 5,000 |
| [Open Images v4]()  | Image | 57 | 3,290,070 | 329 | 374,768 | 9,178,275 |
| [Visual Genome]() | Image | 33,877 | 3,843,636 | 40,480 | 2,347,187 | 108,077 | 
| [VrR-VG]() | Image | 1,600 | 282,460 | 117 | 203,375 | 58,983 |
| [UnRel]() | Image | - | - | 18 | 76 |  1,071 |
| [SpatialSense](*) | Image | 3,679 | - | 9 | 13,229 | 11,569 |
| [SpatialVOC2K]() | Image | 20 | 5,775 | 34 | 9,804 | 2,026 | 
| [ImageNet-VidVRD]() | Video | 35 | - | 132 | 3,219 | 100 |
| [VidOR]() | Video | 80 | - | 50 | - | 10,000 |
| [Action Genome]() | Video | 35 | 0.4M | 25 | 1.7M | 10,000 |
| [FACTUAL]() | Text | - | - | - | - | 40,369 |
</p>


<!-- CVPR-8A2BE2 -->
<!-- NIPS-CD5C5C -->
<!-- ICML-FF7F50 -->
<!-- ICCV-00CED1 -->
<!-- ECCV-1e90ff -->
<!-- TPAMI-BC8F8F -->
<!-- IJCAI-228b22 -->
<!-- AAAI-c71585 -->


# 🍕 Scene Graph Generation

## Image Scene Graph Generation

+ **An Image is Worth 32 Tokens for Reconstruction and Generation** [![Paper](https://img.shields.io/badge/AAAI23-c71585)](https://arxiv.org/abs/2406.07550)<details><summary>Qihang Yu, Mark Weber, Xueqing Deng et al.</summary> Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, Liang-Chieh Chen</details></details>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg)]()
[![citation](https://img.shields.io/badge/citation-0-blue.svg?paper=da0d382c7fa981ba185ca633868442b75cb76de6)](https://www.semanticscholar.org/paper/1e31f4a4ccfc0d1e461be05361d77b5e045f4d37)

## Video Scene Graph Generation


+ **An Image is Worth 32 Tokens for Reconstruction and Generation** [![Paper](https://img.shields.io/badge/AAAI-c71585)](https://arxiv.org/abs/2406.07550)<details><summary>Qihang Yu, Mark Weber, Xueqing Deng et al.</summary> Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, Liang-Chieh Chen</details></details>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg)]()
[![citation](https://img.shields.io/badge/citation-0-blue.svg?paper=da0d382c7fa981ba185ca633868442b75cb76de6)](https://www.semanticscholar.org/paper/1e31f4a4ccfc0d1e461be05361d77b5e045f4d37)



## Audio Scene Graph Generation

+ **An Image is Worth 32 Tokens for Reconstruction and Generation** [![Paper](https://img.shields.io/badge/AAAI-c71585)](https://arxiv.org/abs/2406.07550)<details><summary>Qihang Yu, Mark Weber, Xueqing Deng et al.</summary> Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, Liang-Chieh Chen</details></details>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg)]()
[![citation](https://img.shields.io/badge/citation-0-blue.svg?paper=da0d382c7fa981ba185ca633868442b75cb76de6)](https://www.semanticscholar.org/paper/1e31f4a4ccfc0d1e461be05361d77b5e045f4d37)

## 


# 🥝 Scene Graph Application

## Image Retrieval

+ **Image Retrieval using Scene Graphs** [![Paper](https://img.shields.io/badge/CVPR15-8A2BE2)](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf) ![Star](https://img.shields.io/github/stars/InternLM/InternLM-XComposer.svg?style=social&label=Star) <br> Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, Li Fei-Fei




## Video Retrieval


+ **An Image is Worth 32 Tokens for Reconstruction and Generation** [![Paper](https://img.shields.io/badge/AAAI-c71585)](https://arxiv.org/abs/2406.07550)<details><summary>Qihang Yu, Mark Weber, Xueqing Deng et al.</summary> Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, Liang-Chieh Chen</details></details>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg)]()
[![citation](https://img.shields.io/badge/citation-0-blue.svg?paper=da0d382c7fa981ba185ca633868442b75cb76de6)](https://www.semanticscholar.org/paper/1e31f4a4ccfc0d1e461be05361d77b5e045f4d37)


# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ChocoWu/Awesome-Scene-Graph-for-VL-Learning&type=Date)](https://star-history.com/#ChocoWu/Awesome-Scene-Graph-for-VL-Learning&Date)
