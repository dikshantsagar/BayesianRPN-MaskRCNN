export DETECTRON2_DATASETS=/projects/iiitd/mrcnn/data/
python -W ignore train_net.py --num-gpus 8 --num-machines 2  --machine-rank $1 \
--dist-url "tcp://10.31.229.54:8686" \
--config-file "../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" MODEL.WEIGHTS "/projects/iiitd/mrcnn/pretrained/x101.pkl"
