# RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback

This is the official repository of the paper: [RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback](https://arxiv.org/abs/2403.06840).

![Framework of RA-ISF.](ra-isf.png)

# 

The codebase of RA-ISF integrate iterative question-answering and problem decomposition into the retrieval-augmented generation process. 

Compared to other methods, RA-ISF evaluates whether a question can be answered by assessing the model's capabilities and the relevance of the retrieved texts. When it cannot be answered, the problem decomposition module breaks down the original question into sub-questions for re-evaluation. 

**We will release the code soon.**


## Citation

If you use this codebase, or RA-ISF inspires your work, please cite:

```bibtex 
@misc{liu2024raisf,
      title={RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback}, 
      author={Yanming Liu and Xinyue Peng and Xuhong Zhang and Weihao Liu and Jianwei Yin and Jiannan Cao and Tianyu Du},
      year={2024},
      eprint={2403.06840},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
