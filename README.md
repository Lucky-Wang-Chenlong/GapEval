<div align="center">
<h1>Quantifying the Gap between Understanding and Generation within Unified Multimodal Models</h1>
<a href="https://arxiv.org/abs/2601.13304"><img src="https://img.shields.io/badge/arXiv-2601.13414-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/FrancisChen1/GapEval"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<br>
<br>
<strong>
<a href="https://lucky-wang-chenlong.github.io/">Chenlong Wang<sup>*</sup></a>&nbsp;&nbsp;
<a href="https://yuhangchen1.github.io/">Yuhang Chen<sup>*</sup></a>&nbsp;&nbsp;
<a href="https://openreview.net/profile?id=~Zhihan_Hu1">Zhihan Hu<sup>*</sup></a>&nbsp;&nbsp;
<a href="https://dongping-chen.github.io/">Dongping Chen<sup>1</sup></a>&nbsp;&nbsp;
<a href="https://wenhuchen.github.io/">Wenhu Chen<sup>2</sup></a>&nbsp;&nbsp;
<br>
<a href="https://www.cs.umd.edu/people/sarahwie">Sarah Wiegreffe<sup>1</sup></a>&nbsp;&nbsp;
<a href="https://tianyizhou.github.io/">Tianyi Zhou<sup>3,†</sup></a>&nbsp;&nbsp;
<br>
<br>
<sup>1</sup> University of Maryland &nbsp;&nbsp;
<sup>2</sup> University of Waterloo &nbsp;&nbsp;
<sup>3</sup> MBZUAI
</strong>
</div> 

---

<!-- <img src="figures/intro.png"> -->

We introduce ***GapEval***, a bidirectional benchmark designed to quantify the gap between understanding and generation
capabilities, and quantitatively measure the cognitive coherence of the two “unified” directions.

- The design nature of **GapEval**.

Each question can be answered in both modalities (image and text), enabling a symmetric evaluation of a model’s bidirectional inference capability and cross-modal consistency.

- Future direction for actual unification.

To further explore the underlying mechanism, we conduct an empirical study from the perspective of knowledge manipulation to illustrate the underlying limitations. Our findings indicate that knowledge within UMMs often remains disjoint. The capability emergence and knowledge across modalities are unsynchronized. In this work, we claim that the key of the actual unification lies in the unification of embedded knowledge.

<img src="figures/main-fig.png">

---

## :memo: Contents

- [:memo: Contents](#memo-contents)
- [💡 Updates \& News](#-updates--news)
- [🚀 Evaluation](#-evaluation)
- [⚠️ TODO List](#️-todo-list)
- [👍 Acknowledgement](#-acknowledgement)
- [⭐ Citation](#-citation)

## 💡 Updates & News
- [2026/3] Our paper has been released on Arxiv.
- [2026/3] GapEval dataset is available at `https://huggingface.co/datasets/FrancisChen1/GapEval`.

## 🚀 Evaluation

1. **Load Dataset**
  ```python
  from datasets import load_dataset
  
  repo_id = "FrancisChen1/GapEval"
  dataset = load_dataset(repo_id)
  ```

  Dataset page:
  `https://huggingface.co/datasets/FrancisChen1/GapEval`

2. **Evaluate MLLMs on CausalSpatial**

  Optional models: 
  - GPT5 / GPT-5 mini / Claude / Gemini / Qwen2.5 VL / Qwem3-VL
  ```cli
  python eval.py --model_path GPT5 --output_file ./output-gpt5 --subset collision+physics
  ```
 
3. **Evaluate Gap Between Understanding and Generation**

  Prepare a JSON file in the following format:
  ```json
    {
      "model_name": [
    [p11, p10, p01, p00],
    [p11, p10, p01, p00],
    [p11, p10, p01, p00],
    [p11, p10, p01, p00]
      ]
    }
  ```

  Each row is:
  - `p11`: both tasks are done successfully
  - `p10`: understanding succeeds, generation fails
  - `p01`: generation succeeds, understanding fails
  - `p00`: both tasks fail

  Then run:
  ```bash
  python evaluate.py  /path/to/your/model/test/json/file
  ```

  The script will write a new file in the same folder with the suffix `_gapres.json`.
  Example output:
  ```json
  {
    "results": {
      "model_name": {
        "gaps": [xxx,xxx,xxx,xxx],
        "gap_mean": xxx
      }
    }
  }
  ```

## ⚠️ TODO List
- [x] Dataset Release
- [x] Paper Release


## ⭐ Citation

```
@article{wang2026quantifyinggapunderstandinggeneration,
      title={Quantifying the Gap between Understanding and Generation within Unified Multimodal Models}, 
      author={Chenlong Wang and Yuhang Chen and Zhihan Hu and Dongping Chen and Wenhu Chen and Sarah Wiegreffe and Tianyi Zhou},
      year={2026},
      eprint={2602.02140},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2602.02140}, 
}
```
