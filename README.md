## Run train

```
pip install -r requirements.txt

python train.py --dataset_path /Users/vika/Desktop/eyeglasses_dataset --LR 0.0001 --loss binary_crossentropy --epochs 20 --batch_size 16
```

## Run inference

```
pip install -r requirements.txt

python inference.py --image_path '/Users/vika/Desktop/eyeglasses_dataset/test/images/58080_3.png'
```