export CUDA_VISIBLE_DEVICES=0

for i in 0 1 2 3 4 5 6 7 8 9
do
    python main_train.py --model se_resnext101 --batch_size 24 --fold $i
    python main_train.py --model se_resnext50 --batch_size 24 --fold $i
    python main_train.py --model senet154 --batch_size 12 --fold $i
done