# DeepIRT with independent student and item networks
This is the code for the paper *DeepIRT with independent student and item networks*

Emiko Tsutsumi, Yiming Guo, and Maomi Ueno

If you find the code useful in your research, please cite
```
@inproceedings{EDM2019_DeepIRT,
  title={DeepIRT with independent student and item networks},
  author={Emiko Tsutsumi, Yiming Guo, and Maomi Ueno},
  year={2022},
  booktitle = {},
}
```
## Setups

- Python 3+
- Tensorflow-gpu==1.13.2
- Scikit-learn 0.21.3
- Numpy 1.16.4

## Running the model
Here are some examples for using Deepirt-HN model on ASSISTments2009 dataset:
```
!python main.py --dataset assist2009
```
