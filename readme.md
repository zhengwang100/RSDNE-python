# RSDNE (python light version)
RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-imbalanced Labels for Network Embedding. AAAI18.
This is a shallow method for the problem of [Zero-shot Graph Embedding (ZGE)](https://zhengwang100.github.io/project/zero_shot_graph_embedding.html), i.e., graph embeddings when labeled data cannot cover all classes. 


- This is the python light version of RSNDE. 
- The original code is written in matlab [(code)](https://github.com/zhengwang100/RSDNE).
- More datasets can be found in [(code)](https://github.com/zhengwang100/RECT).


Breifly explain:
---
- RSDNE loss: min_{U,V} J = |G-UV|^2 + alpha*( tr(U'LsU) + tr(U'LwU) ) + lambda(|U|^2 + |V|^2)

- By setting alpha=0 (i.e., removing the relax part), our method will reduce to the common matrix decomposition method, like [MFDW (IJCAI15)](https://www.ijcai.org/Proceedings/15/Papers/299.pdf).


Usage (abstract):
---
- set the dataset
- python main_RSDNE.py


Experiment results: 
---
- label rate 30%:
  - MFDW: (0.5724018973695558, 0.5283697422985723) # set alpha=0 
  - RSDNE: (0.6018602846054334, 0.5766616831506641)

- label rate 50%:
  - MFDW: (0.6144927536231883, 0.5603593000330456) # set alpha=0 
  - RSDNE: (0.6661835748792271, 0.6218905028673257)

Citing
---
If you find RSDNE useful in your research, please cit our paper, thx:
```
@InProceedings{wang2018rsdne,
  title={{RSDNE}: Exploring Relaxed Similarity and Dissimilarity from Completely-imbalanced Labels for Network Embedding},
  author={Wang, Zheng and Ye, Xiaojun and Wang, Chaokun and Wu, YueXin and Wang, Changping and Liang, Kaiwen},
  booktitle={AAAI},
  pages={475--482},
  year={2018}
}
```
