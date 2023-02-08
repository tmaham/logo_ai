import os 

command = "rm -r outputs/*"
os.system(command)

ck = os.listdir("logs")
li = os.path.join("logs",ck[0],"checkpoints","last.ckpt")

prompt = " 'butterfly TT' "
command = "python txt2img.py --ddim_eta 1.0 \
                          --n_samples 5 \
                          --n_iter 5 \
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt " +li +" --prompt" + prompt
os.system(command)

# prompt = " 'lion' "
# command = "python txt2img.py --ddim_eta 1.0 \
#                           --n_samples 10 \
#                           --n_iter 1 \
#                           --ddim_steps 50 \
#                          --scale 5.0\
#                           --ckpt " +li +" --prompt" + prompt
# os.system(command)


# prompt = " 'a photo of an astronaut riding a horse on mars' "
# command = "python txt2img.py --ddim_eta 1.0 \
#                           --n_samples 10 \
#                           --n_iter 1 \
#                           --ddim_steps 50 \
#                          --scale 5.0\
#                           --ckpt " +li +" --prompt" + prompt
# os.system(command)

