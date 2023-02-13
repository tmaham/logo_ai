import os 

name = ["RABBIT"]

black = False
one_font = True


for n in name:
    command = "mkdir data/"+n
    if not os.path.exists("data/"+n):
        os.system(command)

        prompt = f" ' {n} cartoon white background' "
        command = "python txt_gendata.py --ddim_eta 1.0 \
                                --n_samples 25 \
                                --n_iter 2\
                                --ddim_steps 50 \
                                --scale 5.0\
                                --outdir data/"+n+" --ckpt ckpt/model.ckpt --prompt " + prompt
        os.system(command)

    
    command = "rm -r logs/*"
    os.system(command)
    
    command = "python main_finetune.py --base configs/finetune/finetune.yaml \
        -t \
        -n test \
        --gpus 0, \
        --actual_resume ckpt/model.ckpt\
        --data_root data/rabbit " + "--letter "\
            + n + " --style_word " + n + \
        (" --black True" if black else " ") + (" --one_font True" if one_font else " ") + " --images data/" +n 

    os.system(command)

    if black:
        if one_font:
            name_out = "final_outputs_full_black_one"+"/"+n
        else:
            name_out = "final_outputs_full_black"+"/"+n
    else:
        if one_font:
            name_out = "final_outputs_full_one/" + n 
        else:
            name_out = "final_outputs_full/" + n 

    command = "mkdir -p " + name_out
    os.system(command)

    ck = os.listdir("logs")
    li = os.path.join("logs",ck[0],"checkpoints","last.ckpt")

    prompt = f" '{n} logo' "
    command = "python txt2img.py --ddim_eta 1.0 \
                            --n_samples 5 \
                            --n_iter 1 \
                            --ddim_steps 50 \
                            --scale 5.0\
                            --outdir " + name_out + " --ckpt " +li +" --prompt" + prompt
    os.system(command)
