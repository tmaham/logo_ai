
import pixellib
from pixellib.tune_bg import alter_bg


input = "logs/rabbit2023-02-08T11-15-11_test/images/train/samples_scaled_gs-000200_e-000000_b-000200.png"
output = "logs/rabbit2023-02-08T11-15-11_test/images/train/samples_scaled_gs-000200_e-000000_b-000200_white.png"
change_bg = alter_bg()
change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
change_bg.color_bg(input, colors = (0,128,0), output_image_name=output)