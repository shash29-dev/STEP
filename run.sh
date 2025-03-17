
run_title=ref_step

# To train on APT36K
python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
        "pipeline.train=True" "data=aptmmpose_nokpts_kptsemb.yaml" 


# Run Model in validation setting, set pipeline.train=False
# python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=aptmmpose_nokpts_kptsemb.yaml" 


# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=aptmmpose_nokpts_kptsemb.yaml" \
#         "pipeline.video=True" \
#         "pipeline.video_path=/path-to-somewhere/PoseEstimation/AnimalPoseGit/Evaluation/evaldata/videos/C0213.mp4"



#########################################################
##### Configs for other datasets
# run_title=ref_step
# python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
#         "pipeline.train=True" "data=apt10k_nokpts_kptsemb.yaml" 

# run_title=ref_step
# python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
#         "pipeline.train=True" "data=crowdpose_nokpts_kptsemb.yaml" 


# run_title=ref_step
# python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
#         "pipeline.train=True" "data=mousepose_nokpts_kptsemb.yaml" 

# run_title=ref_step
# python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
#         "pipeline.train=True" "data=marmosetpose_nokpts_kptsemb.yaml" 

# run_title=ref_step
# python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=aptmmpose_nokpts_kptsemb.yaml" 

# run_title=ref_step
# python -W ignore main.py config=step_config.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=crowdpose_nokpts_kptsemb.yaml" \
#         "pipeline.awazi_style=False"


# run_title=ref_step
# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=aptmmpose_nokpts_kptsemb.yaml" \
#         "pipeline.video=True" \
#         "pipeline.video_path=./path-to-somewhere/PoseEstimation/AnimalPoseGit/Evaluation/evaldata/continious_frames/vid_022672.mp4"

# run_title=ref_step
# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=crowdpose_nokpts_kptsemb.yaml" \
#         "pipeline.video=True" \
#         "pipeline.video_path=./path-to-somewhere/PoseEstimation/AnimalPoseGit/Evaluation/evaldata/continious_frames/vid_022672.mp4"


# # 3943
# run_title=ref_step
# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=crowdpose_nokpts_kptsemb.yaml" \
#         "pipeline.video=False" \
#         "pipeline.video_path=./ptsw/val/000522_mpii_test" \
#         "pipeline.awazi_style=False"

# run_title=ref_step
# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=mousepose_nokpts_kptsemb.yaml" \
#         "pipeline.video=True" \
#         "pipeline.video_path=/path-to-somewhere/PoseEstimation/AnimalPoseGit/Evaluation/evaldata/videos/mouse.mp4"  \
#         "pipeline.awazi_style=True"


# run_title=ref_step
# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=crowdpose_nokpts_kptsemb.yaml" \
#         "pipeline.video=True" pipeline.awazi_style=True
       

# run_title=ref_step
# fname=apt10k_roll_pd1.3
# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=aptmmpose_soumya_val.yaml" "data.fname=$fname"

# run_title=ref_step
# fname=cp_roll_pd25
# python main.py config=test_pe.yaml "+run_title=$run_title" \
#         "pipeline.train=False" "data=crowdpose_soumya_val.yaml" "data.fname=$fname"