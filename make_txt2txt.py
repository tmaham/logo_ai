import os 

command = "rm -r outputs/*"
os.system(command)

prompt = " 'R in style of rabbit' "
command = "python txt2img.py --ddim_eta 1.0 \
                          --n_samples 5 \
                          --n_iter 1\
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt ckpt/model.ckpt --prompt" + prompt
os.system(command)


prompt = " 'R' "
command = "python txt2img2.py --ddim_eta 1.0 \
                          --n_samples 5 \
                          --n_iter 1\
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt ckpt/model.ckpt --prompt" + prompt
os.system(command)

