
# 🚀 The implementation of paper "Semantic Space Reorganization for Robust Digital Video Semantic Communication"


## 📑 Abstract

```
Quality-of-Service (QoS)-aware semantic communication has emerged as a promising paradigm for next-generation video transmission by prioritizing task-relevant semantics under stringent bandwidth and latency constraints. Among existing approaches, codebook-assisted semantic communication significantly reduces transmission overhead by conveying only the index of a quantized semantic representation for reconstruction at the receiver. However, this index-based paradigm is inherently sensitive to index perturbations induced by unreliable channels, where even minor index deviations may lead to the selection of semantically dissimilar codewords, resulting in pronounced degradation of reconstruction quality and QoS performance.To enhance robustness without increasing system complexity or transmission bitrate, we propose Reordered Residual Stochastic Quantization for robust Digital Video Semantic Communication (RRSQ-DVSC) from the perspective of semantic space organization. Specifically, stochastic residual vector quantization is integrated with a lightweight video backbone to enable efficient and flexible semantic discretization; semantic-aware clustering is applied to the learned codebook, followed by index reordering such that semantically similar codewords occupy adjacent index positions, effectively transforming index perturbations into bounded semantic distortions; and a Semantic-Guided Reconstruction (SGR) module with cross-window fusion attention is introduced to reinforce inter-frame semantic consistency in long video sequences.Experimental results on the HEVC Class B dataset demonstrate that, at an SNR of 1 dB, the proposed RRSQ-DVSC achieves improvements of 4.91\% in PSNR, 5.52\% in MS-SSIM, and 14.25\% in LPIPS over its counterpart without codebook reordering, while incurring negligible latency overhead, validating its effectiveness for robust and QoS-oriented semantic video communication.
```

## 🖥️ Prerequisites

```shell
torch>=2.0
torchvision>=1.2
```

## 🗂️ Test Dataset



## 🗃️ Pre-trained Model 

The model will released soon.

## 📊 Evaluation 

![results](./assets/fingure1.png)


## 📜 Citation 

```
@misc{
    author={},
    title={Semantic Space Reorganization for Robust Digital Video Semantic Communication},
    note={}
}
```


## 🤝 Acknowledgement

Our research is based on [DCVC](https://github.com/microsoft/DCVC) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). Thanks for their excellent work.


<!-- ```
git add .
git commit -m 
git push -u origin main
``` -->

