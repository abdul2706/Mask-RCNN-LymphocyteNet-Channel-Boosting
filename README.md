# Exploitation of Mask-RCNN for the detection of Lymphocytes in histopathological images

## Abstract

Cancer is one of the most commonly occurring deadly diseases around the world. Lymphocytes are considered as an indicator of cancer, which accumulates at the site of tumor regions in response of immune system. Lymphocyte’s detection and quantification play an important role in determining the cancer progression and therapeutic efficacy. However, automation of lymphocyte detection system using Machine learning techniques poses a number of challenges such as unavailability of annotations, sparse representation of lymphocytes, and irregular deposition of stains and presence of artifacts, which gives the false impression of lymphocytes on tissues. Therefore, this project aims to develop an automated detection of lymphocyte in histopathological images. In this regard, the idea of Channel Boosting is used in the backbone of Mask-RCNN to improve its feature extraction ability. Two different backbones are integrated using custom attention blocks and by applying addition operation. An open-source dataset LYSTO is used to evaluate the proposed LymphocyteNet on open-source dataset LYSTO and compared the performance with state-of-the-art architectures. The proposed channel boosted architecture shows improvement in performance in terms of kappa (0.9044), f-score (0.8930), precision (0.8919) and recall (0.8940). The learning capacity of the proposed LymphocyteNet is further evaluated by performing multi-class detection on H&E stained NuCLS dataset. The proposed LymphocyteNet shows good generalization with an f-score of 0.725. The promising result suggests that the idea of channel boosting enhances the learning capacity and can be exploited to improve the detection on complex datasets.

## Dataset

[LYSTO Dataset](https://lysto.grand-challenge.org/ "LYSTO Dataset") was used in this research work.

## Proposed Methadology

![Proposed Architecture](./images/maskrcnn-lymphocytenet3-cm1.jpg)

## Results

|   Backbone of Mask-RCNN    | Recall | Precision | F-Score | Kappa  |
| :------------------------: | :----: | :-------: | :-----: | :----: |
| Proposed LymphocyteNet (+) | 0.8940 |  0.8920   | 0.8930  | 0.9044 |
| Proposed LymphocyteNet (x) | 0.9190 |  0.8530   | 0.8843  | 0.8909 |
|         ResNet-50          | 0.8850 |  0.8650   | 0.8749  | 0.8432 |
|       ResNet-CBAM-50       | 0.8620 |  0.8930   | 0.8777  | 0.8889 |

## How to add new Backbone

TODO

## How to add new Hook

TODO

## How to add new Config

TODO
