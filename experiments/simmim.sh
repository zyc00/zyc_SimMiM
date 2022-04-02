python -m torch.distributed.launch --nproc_per_node 8 main_simmim.py \
--cfg configs/swin_tiny__100ep/simmim_finetune__swin_tiny__img192_window6__100ep.yaml \
--data-path ~/zhouyuchen/imagenet-1000/ILSVRC/Data/CLS-LOC/train \
--batch-size 256 \
--output simmim_base