import os 

command = "rm -r outputs/*"
os.system(command)

ck = os.listdir("logs")
li = os.path.join("logs",ck[0],"checkpoints","last.ckpt")

# prompt = " 'R' "
# command = "python txt2img.py --ddim_eta 1.0 \
#                           --n_samples 10 \
#                           --n_iter 3 \
#                           --ddim_steps 50 \
#                          --scale 5.0\
#                           --ckpt " +li +" --prompt" + prompt
# os.system(command)

prompt = " 'rabbit okramun' "
command = "python txt2img2.py --ddim_eta 1.0 \
                          --n_samples 8 \
                          --n_iter 1\
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt " +li +" --prompt" + prompt
os.system(command)
