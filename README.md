# BiomedJourney

**BiomedJourney: Counterfactual Biomedical Image Generation by Instruction-Learning from Multimodal Patient Journeys**

[Yu Gu*](https://scholar.google.com/citations?user=1PoaURIAAAAJ), [Jianwei Yang*](https://jwyang.github.io/), [Naoto Usuyama](https://www.microsoft.com/en-us/research/people/naotous/), [Chunyuan Li](https://chunyuan.li/), [Sheng Zhang](https://scholar.google.com/citations?user=-LVEXQ8AAAAJ&hl=en), [Matthew P. Lungren](https://aimi.stanford.edu/people/matthew-lungren-0), [Jianfeng Gao](https://scholar.google.com/citations?user=CQ1cqKkAAAAJ&hl=en), [Hoifung Poon](https://scholar.google.com/citations?user=yqqmVbkAAAAJ&hl=en) (* Equal Contribution)


<p align="center">
  <img src="imgs/biomedjourney_teaser_animation.gif" alt="BiomedJourney GIF">
</p>

[![License](https://img.shields.io/badge/Code%20License-Microsoft%20Research-red)]()
**Usage and License Notices**: The model is not intended or made available for clinical use as a medical device, clinical support, diagnostic tool, or other technology intended to be used in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions. The model is not designed or intended to be a substitute for professional medical advice, diagnosis, treatment, or judgment and should not be used as such.  All users are responsible for reviewing the output of the developed model to determine whether the model meets the user’s needs and for validating and evaluating the model before any clinical use. Microsoft does not warrant that the model or any materials provided in connection with will it be sufficient for any medical purposes or meet the health or medical requirements of any person.


## Model Description 

*BiomedJourney* is a novel method for counterfactual medical image generation by instruction-learning from multimodal patient journeys. Given a patient with two medical images taken at different time points, we use GPT-4 to process the corresponding text-based image reports and generate a natural language description of disease progression. The resulting triples (prior image, progression description, new image) are then used to train a latent diffusion model for counterfactual medical image generation. The resulting model substantially outperforms prior state-of-the-art methods in instruction image editing and medical image generation such as IntructPix2Pix and RoentGen.

## Installation

### Prerequisites

Create a new conda environment:
```
conda create -n biomedjourney python=3.9
conda activate biomedjourney
```

Install requirements:

```
pip install -r requirements.txt --user
```

Other dependencies:
```
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers --user
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip --user
pip install git+https://github.com/crowsonkb/k-diffusion.git --user
pip install taming-transformers-rom1504 --user
pip install torchxrayvision --user
pip install scikit-learn --user
```

### Download checkpoints

Download the pretrained stable-diffusion model.
```bash
bash scripts/download_pretrained_sd.sh
```

## Model Training

Training on one node with 8 GPUs:
```bash
python main.py --name {job_name} --base configs/train_biomedjourney_res256.yaml \
    --train --gpus 0,1,2,3,4,5,6,7 --logdir ./model/biomedjourney/{job_name} \
    data.params.train.params.registration=True \
    data.params.batch_size=8 \
    model.params.cond_stage_config.params.max_length=256 \
    model.params.unet_config.params.max_length=256 \
```
In above command, `job_name` is the name of the job. The training log will be saved in `./model/biomedjourney/{job_name}`.

## Model Evaluation

Run gradio interactive demo:
```bash
python biomedjourney_app.py --ckpt {path_to_checkpoint}
```

## Dataset

This model builds upon the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset , which contains 377,110 image-report pairs from 227,827 radiology studies. A patient may have multiple studies, whereas each study may contain multiple chest x-ray (CXR) images taken at different views. In this work, we only use posteroanterior (PA), the standard frontal chest view, and discard AP and lateral views. This results in 157,148 image-text pairs. We follow the standard partition and use the first nine subsets (P10-P18) for training and validation, while reserving the last (P19) for testing. We then identify patient journey and generate 9,354, 1,056, 1,214 counterfactual triples for train, validation, test, respectively. For additional details see the Please refer to [the associated paper](https://arxiv.org/abs/2310.10765).  

## Model Uses 

### Intended Use 

The data, code, and model checkpoints are intended to be used solely for (I) future research on counterfactual medical image generation and (II) reproducibility of the experimental results reported in the reference paper. The data, code, and model checkpoints are not intended to be used in clinical care or for any clinical decision-making purposes.  

### Primary Intended Use 

The primary intended use is to support AI researchers reproducing and building on top of this work. BiomedJourney and its associated models should be helpful for exploring various biomedical counterfactual image generation. 

### Out-of-Scope Use 

**Any** deployed use case of the model --- commercial or otherwise --- is out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are intended *for research use only* and not intended for deployed use cases. 

## Limitations 

This model was developed using English corpora, and thus may be considered English-only. This model is evaluated on a narrow set of biomedical benchmark tasks, described in [BiomedJourney paper](https://arxiv.org/abs/2310.10765). As such, it is not suitable for use in any clinical setting. Under some conditions, the model may make inaccurate predictions and display limitations, which may require additional mitigation strategies. In particular, this model is likely to carry many of the limitations of the models from which it is derived, [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [BiomedCLIP](https://aka.ms/biomedclip). While evaluation has included clinical input, this is not exhaustive; model performance will vary in different settings and is intended for research use only. 

Further, this model was developed in part using the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset. These chest radiographs were collected from Beth Israel Deaconess Medical Center and are thus enriched for the patients receiving care in the surrounding area, a distribution that may not be representative of other sources of biomedical data. Further, this dataset has been shown to contain existing biases characteristic of chest x-ray datasets, e.g. see [Reading Race](https://arxiv.org/pdf/2107.10356.pdf).  

## Citation
```bibtex
@misc{gu2023biomedjourney,
      title={BiomedJourney: Counterfactual Biomedical Image Generation by Instruction-Learning from Multimodal Patient Journeys}, 
      author={Yu Gu and Jianwei Yang and Naoto Usuyama and Chunyuan Li and Sheng Zhang and Matthew P. Lungren and Jianfeng Gao and Hoifung Poon},
      year={2023},
      eprint={2310.10765},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```