# Example script for training
python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-10 --gradient-clip=1
python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-10 --gradient-clip=0

python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-8 --gradient-clip=1
python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-8 --gradient-clip=0

python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-7 --gradient-clip=0
python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-7 --gradient-clip=1

python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-6 --gradient-clip=0
python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-6 --gradient-clip=1

python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-5 --gradient-clip=0
python train.py --epoch=25 --sequence-length=256 --batch-size=8 --lr=5e-5 --gradient-clip=1

