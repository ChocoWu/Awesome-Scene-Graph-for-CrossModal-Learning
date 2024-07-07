<div align="center">
<h1> Awesome-Scene-Graph-for-VL-Learning </h1> 
</div>




# 🎨 Introduction 
A scene graph is a topological structure representing a scene described in text, image, video, or etc. 
In this graph, the nodes correspond to object bounding boxes with their category labels and attributes, while the edges represent the pair-wise relationships between objects. 
  <p align="center">
  <img src="assets/intro1.png" width="75%">
</p>

---

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


---


# 🌷 Scene Graph Datasets
<p align="center">

| Dataset |  Modality  |   Obj. Class  | BBox | Rela. Class | Triplets | Instances | 
|:--------:|:--------:|:--------:| :--------:|  :--------:|  :--------:|  :--------:|
| [Visual Phrase](https://vision.cs.uiuc.edu/phrasal/) | Image | 8 | 3,271 | 9 | 1,796 | 2,769 |
| [Scene Graph](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf) | Image | 266 | 69,009 | 68 | 109,535 | 5,000 |
| [VRD](https://cs.stanford.edu/people/ranjaykrishna/vrd/)  | Image | 100 | - | 70 | 37,993 | 5,000 |
| [Open Images v7](https://storage.googleapis.com/openimages/web/index.html)  | Image | 57 | 3,290,070 | 329 | 374,768 | 9,178,275 |
| [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | Image | 33,877 | 3,843,636 | 40,480 | 2,347,187 | 108,077 | 
| [VrR-VG](http://vrrvg.com/) | Image | 1,600 | 282,460 | 117 | 203,375 | 58,983 |
| [UnRel](https://www.di.ens.fr/willow/research/unrel/) | Image | - | - | 18 | 76 |  1,071 |
| [SpatialSense](https://github.com/princeton-vl/SpatialSense) | Image | 3,679 | - | 9 | 13,229 | 11,569 |
| [SpatialVOC2K](https://github.com/muskata/SpatialVOC2K) | Image | 20 | 5,775 | 34 | 9,804 | 2,026 | 
| [OpenSG](https://github.com/Jingkang50/OpenPSG) | Image (panoptic) | 133 | - | 56 | - | 49K |
| [AUG](https://arxiv.org/pdf/2404.07788) | Image (Overhead View) | 76 | - | 61 | - | - |
| [STAR](https://arxiv.org/pdf/2406.09410) | Satellite Imagery | 48 | 219,120 | 58 | 400,795 | 31,096 |
| [ReCon1M](https://arxiv.org/pdf/2406.06028) | Satellite Imagery | 60 |  859,751 | 64 | 1,149,342 |  21,392 |
| [SkySenseGPT](https://github.com/Luo-Z13/SkySenseGPT) | Satellite Imagery (Instruction) | - | - | - | - | - |
| [ImageNet-VidVRD](https://xdshang.github.io/docs/imagenet-vidvrd.html) | Video | 35 | - | 132 | 3,219 | 100 |
| [VidOR](https://xdshang.github.io/docs/vidor.html) | Video | 80 | - | 50 | - | 10,000 |
| [Action Genome](https://github.com/JingweiJ/ActionGenome) | Video | 35 | 0.4M | 25 | 1.7M | 10,000 |
| [AeroEye](https://arxiv.org/pdf/2406.01029) | Video (Drone-View) | 56 | - | 384 | - | 2.2M |
| [PVSG](https://jingkang50.github.io/PVSG/) | Video (panoptic) | 126 | - |  57 |  - | 400|
| [3D Semantic Scene Graphs (3DSSG)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wald_Learning_3D_Semantic_Scene_Graphs_From_3D_Indoor_Reconstructions_CVPR_2020_paper.pdf) | 3D | 40 | - | - | - | 48K|
| [FACTUAL](https://github.com/zhuang-li/FactualSceneGraph) | Text | - | - | - | - | 40,369 |
</p>


---

<!-- CVPR-8A2BE2 -->
<!-- WACV-6a5acd -->
<!-- NIPS-CD5C5C -->
<!-- ICML-FF7F50 -->
<!-- ICCV-00CED1 -->
<!-- ECCV-1e90ff -->
<!-- TPAMI-BC8F8F -->
<!-- IJCAI-228b22 -->
<!-- AAAI-c71585 -->
<!-- arXiv-b22222 -->


# 🍕 Scene Graph Generation

## 2-D (Image) Scene Graph Generation


### LLM-based 

+ [**Scene Graph Generation Strategy with Co-occurrence Knowledge and Learnable Term Frequency**](https://arxiv.org/pdf/2405.12648) [![Paper](https://img.shields.io/badge/ICML24-FF7F50)]() 

+ [**SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding**](https://arxiv.org/pdf/2406.10100) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Star](https://img.shields.io/github/stars/Luo-Z13/SkySenseGPT.svg?style=social&label=Star)](https://github.com/Luo-Z13/SkySenseGPT) 


+ [**From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models**](https://arxiv.org/pdf/2404.00906) [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Star](https://img.shields.io/github/stars/SHTUPLUS/Pix2Grp_CVPR2024.svg?style=social&label=Star)]() 


### Non-LLM-based

+ [**Leveraging Predicate and Triplet Learning for Scene Graph Generation**](https://arxiv.org/pdf/2406.02038) [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Star](https://img.shields.io/github/stars/jkli1998/DRM.svg?style=social&label=Star)](https://github.com/jkli1998/DRM) 


+ [**STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery**](https://arxiv.org/pdf/2406.09410) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Star](https://img.shields.io/github/stars/Zhuzi24/SGG-ToolKit.svg?style=social&label=Star)](https://github.com/Zhuzi24/SGG-ToolKit) [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://linlin-dev.github.io/project/STAR)

+ [**ReCon1M:A Large-scale Benchmark Dataset for Relation Comprehension in Remote Sensing Imagery**](https://arxiv.org/pdf/2406.06028) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()



## Spatio-Temporal (Video) Scene Graph Generation


### LLM-based 

+ [**Tri-modal Confluence with Temporal Dynamics for Scene Graph Generation in Operating Rooms**](https://arxiv.org/pdf/2404.09231) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()


### Non-LLM-based
+ [**CYCLO: Cyclic Graph Transformer Approach to Multi-Object Relationship Modeling in Aerial Videos**](https://arxiv.org/pdf/2406.01029) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()


+ [**OED: Towards One-stage End-to-End Dynamic Scene Graph Generation**](https://arxiv.org/pdf/2405.16925) [![Paper](https://img.shields.io/badge/CVPR24-8A2BE2)]() [![Star](https://img.shields.io/github/stars/guanw-pku/OED.svg?style=social&label=Star)](https://github.com/guanw-pku/OED)




## Audio Scene Graph Generation

+ [**Image Retrieval using Scene Graphs**](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf) [![Paper](https://img.shields.io/badge/CVPR15-8A2BE2)]() [![Star](https://img.shields.io/github/stars/InternLM/InternLM-XComposer.svg?style=social&label=Star)]()



## 3D Scene Graph Generation

+ [**EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion**](https://arxiv.org/pdf/2405.00915) [![Paper](https://img.shields.io/badge/ECCV24-1e90ff)]() [![Star](https://img.shields.io/github/stars/ymxlzgy/echoscene.svg?style=social&label=Star)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://sites.google.com/view/echoscene)


+ [**Incremental 3D Semantic Scene Graph Prediction from RGB Sequences**](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Incremental_3D_Semantic_Scene_Graph_Prediction_From_RGB_Sequences_CVPR_2023_paper.pdf) [![Paper](https://img.shields.io/badge/CVPR23-8A2BE2)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://shunchengwu.github.io/MonoSSG)

+ [**SceneGraphFusion: Incremental 3D Scene Graph Prediction from RGB-D Sequences**](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_SceneGraphFusion_Incremental_3D_Scene_Graph_Prediction_From_RGB-D_Sequences_CVPR_2021_paper.pdf) [![Paper](https://img.shields.io/badge/CVPR21-8A2BE2)]() [![Star](https://img.shields.io/github/stars/ShunChengWu/SceneGraphFusion.svg?style=social&label=Star)](https://github.com/ShunChengWu/SceneGraphFusion) [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://shunchengwu.github.io/SceneGraphFusion)

+ [**Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wald_Learning_3D_Semantic_Scene_Graphs_From_3D_Indoor_Reconstructions_CVPR_2020_paper.pdf) [![Paper](https://img.shields.io/badge/CVPR20-8A2BE2)]()  [![Star](https://img.shields.io/github/stars/ShunChengWu/3DSSG.svg?style=social&label=Star)](https://github.com/ShunChengWu/3DSSG) 





---




# 🥝 Scene Graph Application




## Image Retrieval

+ [**Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval**](https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_Cross-modal_Scene_Graph_Matching_for_Relationship-aware_Image-Text_Retrieval_WACV_2020_paper.pdf) [![Paper](https://img.shields.io/badge/WACV20-800080)]()


+ [**Image Retrieval using Scene Graphs**](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf) [![Paper](https://img.shields.io/badge/CVPR15-8A2BE2)]()







## Image Generation

+ [**SG-Adapter: Enhancing Text-to-Image Generation with Scene Graph Guidance**](https://arxiv.org/pdf/2405.15321) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()


+ [**Generated Contents Enrichment**](https://arxiv.org/pdf/2405.03650) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()



## Visual Question Answering 
+ [**SOK-Bench: A Situated Video Reasoning Benchmark with Aligned Open-World Knowledge**](https://arxiv.org/pdf/2405.09713) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://bobbywu.com/SOKBench/)




## Mitigate Hallucination 

+ [**BACON: Supercharge Your VLM with Bag-of-Concept Graph to Mitigate Hallucinations**](https://arxiv.org/pdf/2407.03314) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Star](https://img.shields.io/github/stars/ztyang23/BACON.svg?style=social&label=Star)](https://github.com/ztyang23/BACON) [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://ztyang23.github.io/bacon-page/)


## Dynamic Environment Guidance

+ [**Situational Instructions Database: Task Guidance in Dynamic Environments**](https://arxiv.org/pdf/2406.13302) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Star](https://img.shields.io/github/stars/mindgarage/situational-instructions-database.svg?style=social&label=Star)](https://github.com/mindgarage/situational-instructions-database) 


## Privacy-sensitive Object Identification
+ [**Beyond Visual Appearances: Privacy-sensitive Objects Identification via Hybrid Graph Reasoning**](https://arxiv.org/pdf/2406.12736) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()


## Video Retrieval


+ [**A Review and Efficient Implementation of Scene Graph Generation Metricsl**](https://arxiv.org/pdf/2404.09616) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Star](https://img.shields.io/github/stars/lorjul/sgbench.svg?style=social&label=Star)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://lorjul.github.io/sgbench/)



---


# Evaluation Metrics

+ [**A Review and Efficient Implementation of Scene Graph Generation Metrics**](https://arxiv.org/pdf/2404.09616) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]() [![Star](https://img.shields.io/github/stars/lorjul/sgbench.svg?style=social&label=Star)]() [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://lorjul.github.io/sgbench/)

+ [**Semantic Similarity Score for Measuring Visual Similarity at Semantic Level**](https://arxiv.org/pdf/2406.03865) [![Paper](https://img.shields.io/badge/arXiv24-b22222)]()



# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ChocoWu/Awesome-Scene-Graph-for-VL-Learning&type=Date)](https://star-history.com/#ChocoWu/Awesome-Scene-Graph-for-VL-Learning&Date)
