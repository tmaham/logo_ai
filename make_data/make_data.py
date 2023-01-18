import os 
import pdb 



base_word = "rabbit"
output_path = "/localhome/mta122/PycharmProjects/logo_ai/data/custom_data/" + base_word 

command = "mkdir " + output_path
os.system(command)

command = "rm -r " + output_path + "/*"
os.system(command)


li = "sd-v1-1.ckpt"
prompt = base_word
command = "python generate_data.py --ddim_eta 1.0 \
                          --n_samples 50 \
                          --n_iter 1\
                          --ddim_steps 50 \
                         --scale 5.0\
                          --ckpt " +li +" --prompt " + prompt + " --output_path " + output_path
os.system(command)



