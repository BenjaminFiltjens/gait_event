# Gait event
Automated gait event detection for Parkinson's Disease patients with freezing of gait. From our paper:
Paper doi here

## Info
1. Read script to extract the lower limb kinematics from a directory containing c3d files and write them to csv.
2. Training script to generate the input vectors and train one of the deep learning models from tf_models.
3. Labelling script to import a pre-trained model and annotate gait events.

## Unpublished results
The deep learning models, which were trained on 15 freezers, were additionally evaluated on fifteen non-freezers. 

|      | IC (n=304) |       | EC (n=278) |       |
| ---- | :--------: | ----- | :--------: | ----- |
|      |    TCN     | LSTM  |    TCN     | LSTM  |
| TP   |    304     | 304   |    278     | 278   |
| FP   |     1      | 1     |     6      | 3     |
| FN   |     0      | 0     |     0      | 0     |
| F1   |   0.998    | 0.998 |   0.989    | 0.995 |

Bland-Altman plot: 
![](https://github.com/BenjaminFiltjens/gait_event/blob/master/nonFOG-ba.png)

