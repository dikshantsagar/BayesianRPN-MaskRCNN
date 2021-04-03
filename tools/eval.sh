export DETECTRON2_DATASETS=/projects/iiitd/mrcnn/data/
python -W ignore train_net.py --num-gpus 8 --config-file "../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" --eval-only MODEL.WEIGHTS "../tools/output_r101/model_final.pth"
