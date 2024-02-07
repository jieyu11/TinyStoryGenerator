import argparse
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import toml
from time import time
from datetime import timedelta
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFinetune:
    """
    Class for finetuning a GPT-2 (and its variants) model.
    """
    def __init__(self, config):
        """
        Initialization of the ModelFinetune class.
        Parameters:
            config: dict of configuration parameters, the keys are listed.
            - input_file: (str) input file name with the first column being the
            training data.
            - batch_size: (int, optional, default 16) batch size used in
            finetuning.
            - epochs: (int, optional, default 5) number of epochs in training.
            - learning_rate: (float, optional, default 3e-5) the learning rate
            during model training.
            - warmup_steps: (int, optional, default 5e3) the warmup steps
            during model training.
            - max_seq_len: (int, optional, default 500) the maximum allowed
            input text length in number of tokens.
            - model_name: (str, optional, default "gpt2") other options are
            "distilgpt2" and "sshleifer/tiny-gpt2"
            - device: (str, optional, default None) it is either "cpu" or
            "gpu", if it is set then the device is used. If it is not set, then
            the code checks if there are gpus, if so use them, otherwise use
            cpu.
            - model_output: (str) output file name
        """
        self.batch_size = config.get("batch_size", 16)
        self.epochs = config.get("epochs", 5)
        self.lr = config.get("learning_rate", 3e-5)
        self.warmup_steps = config.get("warmup_steps", 5e3)
        logger.info("Finetuning hyper parameters:")
        logger.info("  - batch size: %d" % self.batch_size)
        logger.info("  - n epochs: %d" % self.epochs)
        logger.info("  - learning rate: %f" % self.lr)
        logger.info("  - warm-up steps: %d" % self.warmup_steps)

        self.max_seq_len = config.get("max_seq_len", 500)
        logger.info("Maximum sequence length: %d" % self.max_seq_len)

        self.lora_peft = config.get("lora_peft", False)
        #
        # pre-trained model name: "gpt2", "distilgpt2", "sshleifer/tiny-gpt2"
        #
        self.model_name = config.get("model_name", "gpt2")
        logger.info("Pre-trained model name: %s" % self.model_name)

        if "device" in config:
            self.device = config["device"]
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info("The device for training: %s. " % self.device)

        self.model_output = config.get("model_output", "model_trained.pt")
        assert self.model_output.endswith(".pt"), "use .pt for model output!"
        self.save_per_epoch = config.get("save_per_epoch", False)
        self.model_folder = config.get("model_folder", None)
        if self.lora_peft:
            assert self.model_folder is not None, "Must set a folder for LoRA Peft run."

        # check if the output folder exists, if not, create one!
        outfold = "/".join(self.model_output.split("/")[0:-1])
        if outfold and (not os.path.exists(outfold)):
            os.makedirs(outfold)
            logger.info("Making dir: %s for output!" % outfold)

        try:
            input_file = config.get("input_file", None)
            assert input_file is not None, "Set input_file in config!"
            logger.info("Reading input file: %s" % input_file)
            self.data = pd.read_csv(input_file, lineterminator='\n')
        except Exception:
            logger.error("Dataset is not loaded. Make sure setup input_file!")

    @staticmethod
    def trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def model_lora(self, model):
        LORA_R = 256 # 512
        LORA_ALPHA = 512 # 1024
        LORA_DROPOUT = 0.05
        # Define LoRA Config
        lora_config = LoraConfig(
                         r = LORA_R, # the dimension of the low-rank matrices
                         lora_alpha = LORA_ALPHA, # scaling factor for the weight matrices
                         lora_dropout = LORA_DROPOUT, # dropout probability of the LoRA layers
                         bias="none",
                         task_type="CAUSAL_LM",
                         # target_modules=["query_key_value"],
        )

        # # Prepare int-8 model for training - utility function that prepares a PyTorch model for int8 quantization training. <https://huggingface.co/docs/peft/task_guides/int8-asr>
        # model = prepare_model_for_int8_training(model)
        # initialize the model with the LoRA framework
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("saving model config")
        model.config.to_json_file("adapter_config.json")
        return model

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.lora_peft:
            logger.info("Loading LoRA model.")
            model = self.model_lora(model)
        logger.info("Number of trainable parameters: %d." % self.trainable_parameters(model))
        model = model.to(self.device)
        model.train()
        return model
        
    def finetune(self):
        """
        GPT-2 model fine-tuning.
        """
        logger.info("Training model: %s" % self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = self._load_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_training_steps=-1,
            num_warmup_steps=self.warmup_steps)
        proc_seq_count = 0
        sum_loss = 0.0
        batch_count = 0
        for epoch in range(self.epochs):
            logger.info("Starting epoch %d!" % epoch)
            for idx in range(len(self.data)):
                # assuming first column of the data input contains train data.
                text = self.data.iloc[idx, 0]
                if idx % 1000 == 0 and epoch == 0:
                    logger.info("index %d, input text: %s" % (idx, text))
                tensor = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(self.device)
                # Sequence to model
                outputs = model(tensor, labels=tensor)
                loss, logits = outputs[:2]
                loss.backward()
                sum_loss = sum_loss + loss.detach().data
                proc_seq_count = proc_seq_count + 1
                if proc_seq_count == self.batch_size:
                    proc_seq_count = 0
                    batch_count += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                # tracking the sum of loss by every 100 batches.
                if batch_count == 100:
                    logger.info("Sum loss: %f" % sum_loss)
                    batch_count = 0
                    sum_loss = 0.0
            if self.save_per_epoch and not self.lora_peft:
                outname = self.model_output.replace(
                    ".pt", "_epoch-%d.pt" % epoch)
                logger.info("Saving model state: %s" % outname)
                torch.save(model.state_dict(), outname)

        if self.lora_peft:
            assert self.model_folder is not None, ""
            model.save_pretrained(self.model_folder)
            logger.info("Finetuned model saved in folder: %s" % self.model_folder)
        else:
            logger.info("Finetuned model saved: %s" % self.model_output)
            torch.save(model.state_dict(), self.model_output)


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.toml", type=str,
                        required=True, help="Your config toml file.")

    args = parser.parse_args()
    logger.info("Reading config file: %s" % args.config)
    config = toml.load(args.config)
    t = ModelFinetune(config)
    t.finetune()
    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
