# STEP: Simultaneous Tracking and Estimation of Pose for Animals and Humans

[Project Page](https://shash29-dev.github.io/STEP/)    

## Requirements
Major requirements
- `pytorch=1.13.1`
- `hydra`
- `cv2`
- `pandas`

We exported all packages installed in environment on which this code is tested to run with `conda list -e > requirements.txt` and generated requirements are available at `./requirements.txt`. Please note that these requirements may contain unnecessary libraries unrelated to STEP.


# Datasets

### APT-36K Dataset
 Download the [APT-36K dataset](https://github.com/pandorgan/APT-36K) and put it under ```./datasets/APT-36k``` folder. Full consolidated json ```apt36k_annotations.json``` used for training is available at [Link](https://iitgnacin-my.sharepoint.com/:f:/g/personal/17210095_iitgn_ac_in/EtY-IJh0jtFBmRwn8UCDL-0BcnZrBZXa3_U1PH0kV3g0WQ?e=X49z10). Organise the data in following architecture

        ├── data_root
        │   └── Sequences
        |   :   └── clips
        |   :       └── im1.png, im1.json
        |   :       :
        |   :        └── imn.png, imn.json
        |   :
        |   └── Sequences
        |       └── clips
        |           └── im1.png, im1.json
        |           :
        |           └── imn.png, im2.json
        ├── apt36k_annotations.json


### APT-10K Dataset
 Download the [APT-10K dataset](https://github.com/AlexTheBad/AP-10K) and put it under ```./datasets/APT10k``` folder. Full consolidated json ```ap10k.json``` used for training is available at [Link](https://iitgnacin-my.sharepoint.com/:f:/g/personal/17210095_iitgn_ac_in/EtY-IJh0jtFBmRwn8UCDL-0BcnZrBZXa3_U1PH0kV3g0WQ?e=X49z10). Follow similar setup as APT-36K, and place `.json` in root.

 ### CrowdPose Dataset
 Download the [CrowdPose dataset](https://github.com/jeffffffli/CrowdPose) and put it under ```./datasets/CrowdPose``` folder. Full consolidated json ```ap10k.json``` used for training is available at [Link](https://iitgnacin-my.sharepoint.com/:f:/g/personal/17210095_iitgn_ac_in/EtY-IJh0jtFBmRwn8UCDL-0BcnZrBZXa3_U1PH0kV3g0WQ?e=X49z10). Follow similar setup as APT-36K, and place `.json` in root.


# Training/Evaluation
We provide various settings in `run.sh`. Detailed configs for all datasets and ablation studies are availabe in ```./configs/``` folder.

### Usage of ```run.sh```
- `python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
        "pipeline.train=True" "data=aptmmpose_nokpts_kptsemb.yaml" "
` 
    - `config`: which config to use from `./configs/` folder
    - `pipeline.train`: Sets mode for Inference/Training
    - `data`: change config accordingly at `config/data` with appropriate paths pointing to training datasets, Validation datasets. 


### Pre-trained Weights
Download trained models and consolidated Jsons from [Here](https://iitgnacin-my.sharepoint.com/:f:/g/personal/17210095_iitgn_ac_in/EtY-IJh0jtFBmRwn8UCDL-0BcnZrBZXa3_U1PH0kV3g0WQ?e=X49z10) 

Place downloaded models at as pointed by the key `snaps.model_save_dir` in `config` flag, and `.json` file as mentioned in Dataset section. 

# Running STEP on your Videos
`Coming Soon`


# Acknowledgements
Code in this repository is inspired and use utilities from : [MMPose](https://github.com/open-mmlab/mmpose), [PyTracking](https://github.com/visionml/pytracking). We thank the authors for their amazing works and sharing the code.

# Bib
```

```