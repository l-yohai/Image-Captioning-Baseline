import os

from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    Seq2SeqTrainer,
    default_data_collator,
    AutoTokenizer,
    HfArgumentParser,
)

from arguments import ModelArguments, DataTrainingArguments, CaptionTrainingArguments
from dataset import ImageCaptioningDataset, load_dataset


def set_vision_encoder_decoder_model_config(model_args, tokenizer, vision_encoder_decoder_model):
    vision_encoder_decoder_model.config.decoder_start_token_id = tokenizer.bos_token_id
    vision_encoder_decoder_model.config.pad_token_id = tokenizer.bos_token_id  # <|endoftext|>
    vision_encoder_decoder_model.config.vocab_size = vision_encoder_decoder_model.config.decoder.vocab_size
    vision_encoder_decoder_model.config.eos_token_id = tokenizer.bos_token_id
    vision_encoder_decoder_model.config.max_length = model_args.max_length
    vision_encoder_decoder_model.config.early_stopping = model_args.early_stopping
    vision_encoder_decoder_model.config.no_repeat_ngram_size = model_args.no_repeat_ngram_size
    vision_encoder_decoder_model.config.length_penalty = model_args.length_penalty
    vision_encoder_decoder_model.config.num_beams = model_args.num_beams

    vision_encoder_decoder_model.decoder.resize_token_embeddings(
        len(tokenizer))
    return vision_encoder_decoder_model


def main(args):
    model_args, data_training_args, caption_training_args = args

    vit_feature_extractor = ViTFeatureExtractor.from_pretrained(
        model_args.encoder_model_name_or_path)
    vision_encoder_decoder_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        model_args.encoder_model_name_or_path, model_args.decoder_model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.decoder_model_name_or_path)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    vision_encoder_decoder_model = set_vision_encoder_decoder_model_config(
        model_args=model_args, tokenizer=tokenizer, vision_encoder_decoder_model=vision_encoder_decoder_model)

    train_dataset, val_dataset = load_dataset(
        root_dir=".",
        dataset_path=data_training_args.dataset_path,
        feature_extractor=vit_feature_extractor,
        tokenizer=tokenizer,
        max_target_length=model_args.max_length
    )

    trainer = Seq2SeqTrainer(
        model=vision_encoder_decoder_model,
        tokenizer=tokenizer,
        args=caption_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CaptionTrainingArguments))
    args = parser.parse_args_into_dataclasses()

    main(args)
