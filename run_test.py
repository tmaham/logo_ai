import os 

command = "rm -r outputs/*"
os.system(command)

ck = os.listdir("logs")
li = os.path.join("logs",ck[0],"checkpoints","last.ckpt")

prompt = " 'R' "
command = "python txt2img.py --ddim_eta 1.0 \
                          --n_samples 8 \
                          --n_iter 10\
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt " +li +" --prompt" + prompt
os.system(command)

prompt = " 'R' "
command = "python txt2img.py --ddim_eta 0.0 --plms \
                          --n_samples 8 \
                          --n_iter 10\
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt " +li +" --prompt" + prompt
os.system(command)

# base_word = "rabbit"
# output_path = "/localhome/mta122/PycharmProjects/logo_ai/data/custom_data/" + base_word 

# li = "ckpt/model.ckpt"
# prompt = " 'rabbit' "
# command = "python txt2img.py --ddim_eta 0.0 \
#                           --n_samples 50 \
#                           --n_iter 1\
#                           --ddim_steps 50 \
#                          --scale 5.0\
#                           --ckpt " +li +" --prompt " + prompt  + " --plms" + " --output_path " + output_path
# os.system(command)
