import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in list(range(torch.cuda.device_count())))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import hydra
import sys
from omegaconf import OmegaConf
import pdb
import eval_apmm, eval_video
import train_ap


def main(cfg: OmegaConf) -> None:
    if cfg.pipeline.train==True:
        os.makedirs(cfg.snaps.image_save_dir,exist_ok=True)
        os.makedirs(cfg.snaps.model_save_dir,exist_ok=True)
        os.makedirs(cfg.snaps.model_save_dir_bkp,exist_ok=True)
        train_ap.training_loop(cfg)
    else:
        if cfg.pipeline.video==True and cfg.pipeline.awazi_style==False:
            eval_video.evaluation_loop(cfg)
        elif cfg.pipeline.video==True and cfg.pipeline.awazi_style==True:
            # eval_awazi.evaluation_loop(cfg)
            pass
        else:
            eval_apmm.evaluation_loop(cfg)
            pass


if __name__ == "__main__":
    config_path = "./configs/"
    if len(sys.argv) > 1 and sys.argv[1].startswith("config="):
        config_name = sys.argv[1].split("=")[-1]
        sys.argv.pop(1)
    else :
        config_name='test.yaml'

    main_wrapper = hydra.main(config_path=config_path, config_name=config_name,version_base=None)
    main_wrapper(main)()