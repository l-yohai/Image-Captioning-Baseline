# Image Captioning Baseline
Image Captioning Baseline with VisionEncoderDecoderModel in transformers(huggingface)

## Dirs
```
.
├── arguments.py # arguments for training
├── dataset.py # pytorch datasets
└── train.py
```

## Dataset

- Modifying `dataset.py`
- Baseline is fitted with MSCOCO dataset. The JSON file have two columns, "captions" and "file_path".

## Usage

```python
python train.py \
    --encoder_model_name_or_path "google/vit-base-patch16-224-in21k" \
    --decoder_model_name_or_path "gpt2" \
    --max_length 512 \
    --no_repeat_ngram_size 3\
    --length_penalty 2.0\
    --num_beams 4\
    --dataset_path "Captioning Dataset PATH"
    --do_train \
    --do_eval
```

## Arguments

- Refer `arguments.py`, with huggingface Seq2Seq Arguments and Model, Data arguments classes.