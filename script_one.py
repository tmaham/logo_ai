import os 
import pdb 

n="MUSIC"
l = "C"
black = False
one_font = False
just_gen = False

isExist = os.path.exists("data/"+n)

if not isExist:
    command = "mkdir data/"+n
    os.system(command)

    prompt = f" ' {n} cartoon white background' "
    command = "python txt_gendata.py --ddim_eta 1.0 \
                            --n_samples 25 \
                            --n_iter 1\
                            --ddim_steps 50 \
                            --scale 5.0\
                            --outdir data/"+n+" --ckpt ckpt/model.ckpt --prompt " + prompt
    os.system(command)


command = "rm -r logs/*"
if not just_gen: os.system(command)

command = "python main_finetune.py --base configs/finetune/finetune.yaml \
            -t \
            -n test \
            --gpus 0, \
            --actual_resume ckpt/model.ckpt\
            --data_root data/rabbit " + "--letter "\
             + l + " --style_word " + n + \
            (" --black True" if black else " ") + (" --one_font True" if one_font else " ") + " --images data/" +n 

if not just_gen: os.system(command)

# 
if black:
    if one_font:
        name_out = "final_outputs/one/black/"+n+"/"+l
    else:
        name_out = "final_outputs/all/black/"+n+"/"+l
else:
    if one_font:
        name_out = "final_outputs/one/color/" + n + "/" +l
    else:
        name_out = "final_outputs/all/color/" + n + "/" +l


command = "rm -r " + name_out
if not just_gen: os.system(command)

command = "mkdir -p " + name_out
if not just_gen: os.system(command)

ck = os.listdir("logs")
li = os.path.join("logs",ck[0],"checkpoints","last.ckpt")

prompt = f" '{n} {l}' "
command = "python txt2img.py --ddim_eta 1.0 \
                        --n_samples 25 \
                        --n_iter 2 \
                        --ddim_steps 50 \
                        --scale 5.0\
                        --outdir " + name_out + " --ckpt " +li +" --prompt" + prompt
os.system(command)

letter = l
name = n
mode = "one" if one_font else "all"
color = "black" if black else "color"

command = f"python ranking/main.py {name} {letter} {mode} {color}"
os.system(command)