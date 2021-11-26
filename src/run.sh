# Example script for training
python train.py --cuda=True --pretrained-model=bertinho-gl-base-cased --freeze-bert=False --lstm-dim=-1 --language=galician --seed=1 --lr=5e-6 --epoch=20 --use-crf=False --augment-type=none  --augment-rate=0.15 --alpha-sub=0.4 --alpha-del=0.4 --data-path=../data/ --save-path=out
