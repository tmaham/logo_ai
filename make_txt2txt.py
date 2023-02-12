import os 
import sys

name = "kangaroo"

command = "mkdir data/"+name
os.system(command)

# command = "rm -r data/kangaroo/*"
# os.system(command)

# prompt = sys.argv[1]
prompt = f" ' {name} cartoon with white background' "
command = "python txt_gendata.py --ddim_eta 1.0 \
                          --n_samples 25 \
                          --n_iter 2\
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt ckpt/model.ckpt --prompt " + prompt
os.system(command)



# prompt = " 'R' "
# command = "python txt2img2.py --ddim_eta 1.0 \
#                           --n_samples 5 \
#                           --n_iter 1\
#                           --ddim_steps 50 \
#                          --scale 5.0\
#                           --ckpt ckpt/model.ckpt --prompt" + prompt
# os.system(command)

