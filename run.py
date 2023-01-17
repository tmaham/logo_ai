import os

# sd-v1-1

command = "rm -r logs/*"
os.system(command)
command = "python main_finetune.py --base configs/finetune/finetune.yaml \
            -t \
            -n rabbit1 \
            --gpus 0, \
            --actual_resume ckpt/sd-v1-1.ckpt\
            --data_root data/rabbit \
            --init_word 'rabbit'"
os.system(command)


