# CoCoFormer
## Introduction
This is the source code of CoCoFormer A controllable feature-rich polyphonic music generation method, trained on JSF dataset.
For more details, see our paper: [CoCoFormer](https://arxiv.org/abs/2310.09843)

Polyphonic music is a unique form of music with different melodies. However, the same melody often has the same musical texture. We don't have different musical textures for one song to choose from. Therefore, we want to use a music AI model to help us compose polyphonic music. If we have a model, we can specify a melody and a texture and the model can generate a unique polyphonic music. The CoCoFormer is aimed at this.

## Install Dependencies
python 3.7.13  
pytorch 1.8.1+cu101  
mido 1.2.10  
tqdm 4.64.0  

Other third party libraries can be installed using pip install 

## How to use
Tokenize: build event of event2word.pkl and word2event.pkl
```
cd CoCoFormer/dataset/JSF_dataset
python build_jsf.py
```

Dataset processing
```
cd CoCoFormer/dataset
python preprocess_jsf.py
```

Train the model:
```
python train_jsf.py --rpr 
```

Generate polyphonic music:
Conditional generate: input a melody, beat and chord, CoCoFormer will create music according to the specific token.
unconditional generate: create a polyphonic music with no specific token.

As the code shown, we give an example of beat and chord with a specific melody, you can use different function to chose generate ways. 
```
python generate.py
```

### Demo
Under Construction

## Contributing
Thanks for gwinndr, we start our research with [MusicTransformer](https://github.com/gwinndr/MusicTransformer-Pytorch)
Thanks for [Tengfei Niu](https://github.com/fleetingtime1), he implemented some of these functions and network building.
