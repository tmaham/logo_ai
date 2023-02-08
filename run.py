import os


command = "rm -r logs/*"
os.system(command)
command = "rm -r output_log/*"
os.system(command)

command = "python main_finetune.py --base configs/finetune/finetune.yaml \
            -t \
            -n test \
            --gpus 0, \
            --actual_resume ckpt/model.ckpt\
            --data_root data/rabbit" 

os.system(command)


# command = "python scripts/txt2img.py --prompt 'rabbit' --ckpt ckpt/sd-v1-1.ckpt"
# os.system(command)

