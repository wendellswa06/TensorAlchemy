<div align="center">

![EnhancedClipRewardModel - Header](./contrib/EnhancedClipRewardModel-header.png)

# **EnhancedClipRewardModel: Advanced Image-Text Similarity Scoring**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
</div>

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)

## Overview

The EnhancedClipRewardModel is a sophisticated image-text similarity scoring system built on top of OpenAI's CLIP (Contrastive Language-Image Pre-training) model. This implementation significantly enhances CLIP's capabilities by breaking down complex prompts into individual elements, allowing for a more granular and accurate similarity assessment between text prompts and images.

## How It Works

1. **Prompt Breakdown**: The input prompt is decomposed into individual elements using an LLM.
2. **CLIP Processing**: Each "pixel" or element is processed through CLIP along with the corresponding image.
3. **Similarity Computation**: Similarity scores are computed for each prompt element-image pair.
4. **Score Adjustment**: Scores are normalized and adjusted based on predefined thresholds.
5. **Aggregation**: A final aggregate score is calculated, taking into account all prompt elements.

