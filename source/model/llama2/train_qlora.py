import argparse
import os
import time

import torch
import transformers
from transformers import GenerationConfig, Trainer, set_seed

from chatllms.configs import (DataArguments, GenerationArguments,
                              LoraArguments, ModelArguments, QuantArguments,
                              TrainingArguments)
from chatllms.data import make_supervised_data_module
from chatllms.model import (MMLUEvalCallback, SampleGenerateCallback,
                            SavePeftModelCallback, load_model_tokenizer)
from chatllms.train.training import train_and_evaluate
from chatllms.utils.logger_utils import get_root_logger
from chatllms.utils.model_utils import (check_training_finished,
                                        print_trainable_parameters,
                                        verify_dtypes)

torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments,
         QuantArguments, GenerationArguments))
    (model_args, data_args, training_args, lora_args, quant_args,
     generation_args) = parser.parse_args_into_dataclasses()
    # Check arguments (do not check finetuning_args since it may be loaded from checkpoints)
    data_args.init_for_training()
    training_args.generation_config = GenerationConfig(**vars(generation_args))

    args = argparse.Namespace(**vars(model_args), **vars(data_args),
                              **vars(training_args), **vars(lora_args),
                              **vars(quant_args))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    # Log on each process the small summary:
    logger.info(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        +
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info('Training/evaluation parameters %s', args)
    # Check if training was already completed.
    checkpoint_dir, completed_training = check_training_finished(args, logger)
    args.resume_checkpoint = checkpoint_dir

    # load model and tokenizer
    model, tokenizer = load_model_tokenizer(
        args=args,
        checkpoint_dir=checkpoint_dir,
        is_trainable=args.do_train,
        logger=logger,
    )
    logger.info('Loaded model...')

    logger.info('Printing trainable parameters...')
    print_trainable_parameters(args, model)

    set_seed(args.seed)

    # Verify dtypes
    logger.info('Verifying dtypes...')
    verify_dtypes(model)

    data_module = make_supervised_data_module(tokenizer=tokenizer, args=args)
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      **data_module)
    # Add callback to save adapter model.
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Add callback to generate samples.
    if args.sample_generate:
        trainer.add_callback(
            SampleGenerateCallback(
                tokenizer=tokenizer,
                generation_config=GenerationConfig(**vars(generation_args)),
                logger=logger,
            ))

    if args.do_mmlu_eval:
        eval_callback = MMLUEvalCallback(
            trainer=trainer,
            tokenizer=tokenizer,
            data_dir='./data',
            args=args,
        )
        trainer.add_callback(eval_callback)

    assert args.do_train or args.do_eval or args.do_predict
    if args.do_train or args.do_eval:
        train_and_evaluate(trainer, args, logger)


if __name__ == '__main__':
    main()
