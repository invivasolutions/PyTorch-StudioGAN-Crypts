wandb login c2c840dbd3cd98dd70e8df69cbfe5b0352c10d00
CUDA_VISIBLE_DEVICES=0,1 python3 main.py -t -metrics is fid -cfg configs/AFHQv2/StyleGAN3-r-paper-crypts.yaml -data dataset512/ -save /workspace/weights_out/


ImportError: /root/.cache/torch_extensions/py310_cu118/filtered_lrelu_plugin/2e9606d7cf844ec44b9f500eaacd35c0-nvidia-a40/filtered_lrelu_plugin.so: cannot open shared object file: No such file or directory

