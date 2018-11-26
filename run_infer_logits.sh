export CUDA_VISIBLE_DEVICES=0

FOLDS="0,1,2,3,4,5,6,7,8,9"

python main_infer_logits.py --model se_resnext101 --batch_size 12 --folds $FOLDS
python main_infer_logits.py --model se_resnext50 --batch_size 12 --folds $FOLDS
python main_infer_logits.py --model senet154 --batch_size 6 --folds $FOLDS
