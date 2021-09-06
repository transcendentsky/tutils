## Introduction

(1) Background:  As is well known, segmentation/registration/detection aims to predict a label for xxx.
(2) Recent works and the drawbacks:
(3) key problem and feasible thoughts to solve
(4) Our method and advantage to the problem
(5) Our contribution: are summarized as follows:

## Related work

## Method

% In this section, we introduce the proposed XXX Network in detail. First, we describe the overall structure of XXX in subsection 3.1. Then we expound the proposed RecoveryXX MOdule, X module, and X module in subsection 3.2, 3.3 and 3.4, respectively. Finally, we present the loss function used in the proposed network in subsection 3.5.

% In this section, we first introduce the mathematical details of the training and inference stage of the self-supervised learning part of CC2D in Section~\ref{Sec:stage_1}, respectively. Then, we illustrate how to train a new landmark detector from scratch with pseudo-labels in Section~\ref{Sec:Stage_2}, which are the predictions of CC2D-SSL on the training set. The resulting detector is used to predict results for the test set.

% In this section, we first introduce the mathematical details, and then the details

#### Overall Structure

% The overall structure is shown in Fig.2 , The network can be divided into two parts: (1) base network and (2) texture extraction branch. For base network, we utilize  resnet followed with x, For texture branch, we first extract features from xxx, and then xxx. Finally we concatenate output features from base network and xxx.

#### Module 1

Insight，

1. Our goal is to , ....., so we try to do ...
2. % (why we propose this module?) Due to the nowadays existing problem , Therefore, we propose a module to solve such a problem by xxx
3. % (Compare with related work) Our idea is inspired by xxx, but previous work cannot do this, so we modified/add the function/xxx

#### Module 2

% Some problem ,
% Therefore we propose a module xxx

#### Module 3

Insight

% To further improve module 2, xxx

## Experiments

#### Datasets

% We evaluate our methods on three popular datasets, including ...
% Dataset 1: intro
% Dataset 2:

#### Settings

#### Implementationo Details

% We use resnet as our backbone [1]. Following [33], we set $\alpha=1$, xxx
% We apply Adam as our optimizer, weight decay=0.001, xxx, and all of our experiments are trained on RTX.

#### Analysis

% TODO:
% Some traits in robustness feature:
%     Robustness is associated with Datasets, vary from datasets,
%         % if the distribution of dataset is closer to the real world, the
%     easy-to-learn ?
