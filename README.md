# Learning graph representations of the dancing body

## Dependencies
- torch == 1.4.0
- torch-geometric == 1.6.1
- numpy
- tqdm
- jupyter
- matplotlib
- ffmpeg (recommended for making animations)

## Usage
Example training command to test your setup:

```bash
python train.py --name="test" --epochs=3 --batch_size=1 --batch_limit=1 --node_embedding_dim=36 --seq_len=10 --predicted_timesteps=2
```

Calling the training script automatically loads the raw data contained in the numpy arrays saved in the `data` folder and converts it into graph structures for training. A new folder with the name you gave it ("test", in this case) will be created under the `logs` folder with dataloader objects (`dataloader_*.pth`) saved alongside the log file itself (`log.txt`) and your script's arguments (`args.pkl`).

## Animations 
Predictions on a single test batch:
![](data/animations/test_batch.gif)

Predictions on a test set of 200 timesteps:
![](data/animations/test_200.gif)

Edge type #0 (non-edge):
![](data/animations/edgetype0.gif)

Edge type #1:
![](data/animations/edgetype1.gif)

Edge type #2:
![](data/animations/edgetype2.gif)

Edge type #3:
![](data/animations/edgetype3.gif)