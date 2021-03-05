# **Intro**

# Intro
## Background/Problem:
- Such matching `criteria` are mostly `agnostic` of regions.

- It plays a `crucial` role `in` helping radiologists to understand...

- Accurate and reliable anatomical landmark detection is a `fundamental first step` in **`therapy planning and intervention`**, thus it has attracted great interest from `acdamia and industry`.

- It has been proved crucial in many `medical clinical scenarios` such knee joint surgery, bone age estimation, carotid artery bifurcation, orthognathic and maxillofacial surgeries amd pelvic trauma surgery.

- `As such` (therefore), the registration of the specific `tissues` are more importatnt than other `tissues`.

- The `success` of deep learning methods `relies on` the `availability` of a large number of datasets with annatations; 

- however, `curating` such datasets is `burdensome`, especially for medical images. 

- **`A common wisdom`** is that a model with a better generalization is learned from more data. 

- which might `restrict further application in clinical scenarios`

- `Nevertheless`

- 

## Idea/Method:
- To `relieve such a burden` for landmark detection task, we explore the `feasibility` of using ongly a single annotated image and ...

- CC2D-SSL `captures` the `consistent anatomical information` in a `coarse-to-fine fashion` by comparing the cascade feature representations and ...


- Motivated by this, we `hereby` propose a segmentation-assisted registration network to `emphasize` the ROIs `more`.

- Therefore, we apply the same `heuristic` to `pay more attention` to specific ROIs...

- Our attempt is to improve the registration quality, `boosted` by `the addition of` a few segmentation labels.

- Training the `above-mentioned` segmentation module requires manual labels, which are difficult and `costly` to obtain `in practice`

- To `tackle` the challenge, we propose ...

- Our design is `further inspired` by our observation learned `through/via interactions with` **`clinicians`** that they firstly roughly locate the target regions through a `coarse screening` and then progressively refine the fine-grained location.

- This step `brings two benefits`. `On one hand`, the inference procedure becomes more concise. Om the other hand, recent findings show that 

- 

# Related work
- waiting for more

# Method

- we first `introduce` the `mathematical details` of training and inference stage of ...

- Then we `illustrate` how to train a new landmark detector from scratch. The `resulting` detector is used to predict results for the testset.

- The feature extractor E_r and E_p `project` the input X_r and the patch X_p into `a multi-scale feature space`, resulting in a cascade of embeddings. 

- `guided by` its correspinding coordinates

- the deepest one `differentiates` the target coarse-grained area from every different area in the whole map, 

- `As we aim at` increasing the similarity of the correct pixels while decreasing others, 

- `Correspondingly`, we set the ground truth GT^i of each layer: 

- the `detailed illustration` can be found in the supplemental materials. 

- We utilze the multi-task U-Net $g$ as our backbone, which predicts both heatmap and offset maps `simultaneously` and has `satisfactory` performance. 

- $sign(\cdot)$ is a `sign function` which is used to make sure that ...



# Experiment

- This study uses a `widely-used public dataset` for ... `provided` in/by

- `Following the official challenge`, we use mean radial error (MRE) to measure the Euclidean distance between ...

- All of our models are implemented in Pytorch, `accelarated` by an NVIDIA RTX3090 GPU. 

- takes about 6 hours to converge 