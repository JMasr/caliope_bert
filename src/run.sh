# Example script for training
python train.py --epoch=100 --sequence-length=256 --batch-size=8 --lr=5e-7 --gradient-clip=0 --loss-w=0 --language='es' --pretrained-model='berto' --cuda=0
python train.py --epoch=100 --sequence-length=256 --batch-size=8 --lr=5e-7 --gradient-clip=0 --loss-w=0 --language='gl' --pretrained-model='bertinho' --cuda=1
python train.py --epoch=100 --sequence-length=256 --batch-size=8 --lr=5e-7 --gradient-clip=0 --loss-w=0 --language='en' --pretrained-model='roberta-base' --cuda=2

python train.py --epoch=100 --sequence-length=256 --batch-size=8 --lr=5e-7 --gradient-clip=0 --loss-w=0 --language='gl' --pretrained-model='bert-base-multilingual-uncased' --cuda=3
python train.py --epoch=100 --sequence-length=256 --batch-size=8 --lr=5e-7 --gradient-clip=0 --loss-w=0 --language='es' --pretrained-model='bert-base-multilingual-uncased' --cuda=4