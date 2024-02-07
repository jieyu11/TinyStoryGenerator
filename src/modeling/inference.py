from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging
import argparse
import toml
from time import time
from datetime import timedelta
from random import choice
from peft import LoraConfig, PeftModel
import pandas as pd

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Inference:
    """
    Class to make inferences to the finetuned GPT-2 name generation model.
    """
    def __init__(self, config):
        """
        Initialization of the Inference class.
        Parameters:
            config: dict of configuration parameters, the keys are listed.
            - model_file: (str) fine-tuned model file.
            - model_name: (str) pre-trained model name. It should match the
            fine-tuned model file. Possible values are "gpt2", "distilgpt2" and
            "sshleifer/tiny-gpt2"
            - device: (str, optional, default "cpu") either "gpu" or "cpu"
            - quantized: (bool, optional, default False) if true, quantized
            version of the corresponding model is used.
            - max_generated_len: (int, optional, default 10) the maximum length
            of the generated text.
            - starter: (str, optional, default "[-START-]") the starting string
            to be placed at the beginning of a training text.
            - ender: (str, optional, "[-END-]") the ending of a training text.
            - hyper-parameters: (dict, optional) it changes the
            hyper-parameters while running the inference. If it is not set the
            default values are found through:
                from src.models.inference import Inference
                print(Inference._set_hyper_parameters.__doc__)
            - random_seed: (int, optional, default 13572468) seed to control
            randomness in torch.
        """
        self.output_file = config.get("output_file", "./out.csv")
        self.model_file = config.get("model_file", None)
        self.model_name = config.get("model_name", "distilgpt2")
        self.quantized = config.get("quantized", False)
        self.lora_peft = config.get("lora_peft", None)
        self.model_folder = config.get("model_folder", None)
        self.device = config.get("device", "cpu")

        assert not (self.model_file is None and self.model_folder is None), \
            "model_file and model_folder cannot both be None"

        self.model = self._load_model()
        self.tokenizer = self._setup_tokenizer()

        self.max_generated_len = config.get("max_generated_len", 500)
        logger.info("Max generated length is %d" % self.max_generated_len)

        self.starter = config.get("starter", "<|startoftext|>")
        self.ender = config.get("ender", "<|endoftext|>")
        if "hyper-parameters" in config:
            logger.info("Setting up hyper-parameters")
            self._set_hyper_parameters(config["hyper-parameters"])
        else:
            self._set_hyper_parameters()
            logger.info("No hyper-parameters setup. Use defaults!")

        # random seed to control the outputs
        random_seed = config.get("random_seed", 13572468)
        logger.info("Using random seed: %d" % random_seed)
        torch.manual_seed(random_seed)

    def _load_model(self):
        """
        Load model and tokenizer for inference.
        Parameters:
            model_file: (str) fine-tuned model file.
            model_name: (str) pre-trained model name. It should match the
                fine-tuned model file. Possible values are "gpt2", "distilgpt2" and
                "sshleifer/tiny-gpt2"
            device: (str, optional, default "cpu") either "gpu" or "cpu"
            quantized: (bool, optional, default False) if true, quantized
            version of the corresponding model is used.
        """
        # load the pre-trained GPT-2 model and tokenizer.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, return_dict=True
        )
        if self.lora_peft and os.path.exists(self.model_folder):
            assert os.path.exists(self.model_folder), "Folder %s not exist!" % self.model_folder
            model = PeftModel.from_pretrained(model,  self.model_folder)
            logger.info("Loading model from folder: %s" % self.model_folder)
        else:
            assert os.path.exists(self.model_file), "File %s not exist!" % self.model_file
            model.load_state_dict(
                torch.load(self.model_file, map_location=torch.device(self.device))
            )
            logger.info("Loading model from file: %s" % self.model_file)

        # transfer the model weights to the correct device
        model = model.to(self.device)
        if self.quantized:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
            model.to(self.model.device)
            logger.info("Loading quantized model.")
        logger.info("The model device is: %s" % model.device)
        model.eval()
        return model

    def _setup_tokenizer(self):
        # setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #
        # When generating texts, the logits of right-most token are used to
        # predict the next token, therefore the padding should be on the left.
        #
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _set_hyper_parameters(self, config={}):
        """
        Hyper parameters list for model.generate() function to control the
        output text.

        The notes for the variables are found:
            https://huggingface.co/transformers/main_classes/model.html

        Parameters: config (type: dict) it should contain the following:
            temperature (float > 0, defaults to 1.0)
                The value used to module the next token probabilities. The
                higher it is, more random it can be.
            top_k (int > 0, defaults to 50)
                The number of highest probability vocabulary tokens to keep for
                top-k-filtering.
            top_p (float in (0.0, 1.0] defaults to 1.0)
                If set to float < 1, only the most probable tokens with
                probabilities that add up to top_p or higher are kept for
                generation.
            repetition_penalty (float, optional, defaults to 1.0)
                The parameter for repetition penalty. 1.0 means no penalty. The
                higher it is the less likely the words would repeat themselves.
                See this paper (https://arxiv.org/pdf/1909.05858.pdf) for more
                details.
            length_penalty (float, defaults to 1.0)
                Exponential penalty to the length. 1.0 means no penalty. Set to
                values < 1.0 in order to encourage the model to generate
                shorter sequences, to a value > 1.0 in order to encourage the
                model to produce longer sequences.
            no_repeat_ngram_size (int, defaults to 0)
                If set to int > 0, all ngrams of that size can only occur once.
            num_beam_groups (int, defaults to 1)
                Number of groups to divide num_beams into in order to ensure
                diversity among different groups of beams. This paper for more
                details: https://arxiv.org/pdf/1610.02424.pdf.
            diversity_penalty (float, defaults to 0.0)
                This value is subtracted from a beam's score if it generates a
                token same as any beam from other group at a particular time.
                Note that diversity_penalty is only effective if group beam
                search is enabled.
        """
        # hyper parameters
        self.temperature = config.get("temperature", 1.0)
        self.top_k = config.get("top_k", 50)
        self.top_p = config.get("top_p", 1.0)
        assert self.top_p > 0 and self.top_p <= 1.0, "top_p in [0, 1]"
        self.repetition_penalty = config.get("repetition_penalty", 1.0)
        self.length_penalty = config.get("length_penalty", 1)
        # int If set to int > 0, all ngrams of size no_repeat_ngram_size can
        # only occur once.
        self.no_repeat_ngram_size = config.get("no_repeat_ngram_size", 0)
        self.num_beam_groups = config.get("num_beam_groups", 1)
        self.diversity_penalty = config.get("diversity_penalty", 0.0)
        self.bad_words = config.get("bad_words", None)

    def _clean_output(self, output):
        """
        Read the output string and extract the business name.
        Parameters:
            output: output texts. 
        Returns:
            The extracted business names: "apple producing" and "apple
            producing and corp" in the two examples above are returned.
        """
        if self.ender in output:
            # if ender is part of the text, then return text before ender.
            text = output.split(self.ender)[0]
        else:
            text = output
            # if ender is not found. Check if it is partially there.
            # if so, remove it. If none of ender is found, then return the
            # last element after separator.
            endtoremove = self.ender[0:-1]
            while len(endtoremove) > 0:
                if text.endswith(endtoremove):
                    text = text[0:-len(endtoremove)]
                    break
                # strip off the last character.
                endtoremove = endtoremove[0:-1]
        text = text[len(self.starter)+1:]
        return text

    def reset_random_seed(self, random_seed):
        """
        Reset random seed for name generation.
        """
        if not isinstance(random_seed, int):
            logger.warning("Re-set random seed with integer.")
            return None

        logger.info("Re-setting random seed to: %d" % random_seed)
        torch.manual_seed(random_seed)

    def generate(self, data):
        """
        Given a list of data input texts, generate a list of business names
        for each input item.
        Parameters:
            data: (type: list of str) it contains a list of queries texts.
            batchsize: how many of the queries from the input are processed as
            a batch.
        Returns:
            A list of output
        """
        inputs = self.tokenizer(data, return_tensors="pt", padding=True)
        badwords = None
        if self.bad_words is not None:
            # , add_prefix_space=True
            badwords = self.tokenizer(self.bad_words)

        output_sequences = self.model.generate(
            input_ids=inputs['input_ids'].to(self.model.device),
            attention_mask=inputs['attention_mask'].to(self.model.device),
            # disable sampling to test if batching affects output
            # do_sample=False,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            # let it generate longer than original input length
            max_length=len(inputs['input_ids'][0])+self.max_generated_len,
            # the num_return_sequences is set to 1, since data is the repeated
            # input texts.
            num_return_sequences=1,
            bad_words_ids=badwords.input_ids
        )
        outputs = [self.tokenizer.decode(x) for x in output_sequences]
        outputs = [self._clean_output(v) for v in outputs]
        return outputs

    def _input_data_prep(self, query, nresults):
        """ Prepare the input data.

        Args:
            query (type: str): user input text input
            nresults (type: int): number of results to generate
        """
        # data is a list of size nresults
        data = []
        for i in range(nresults):
            text = "%s %s" % (self.starter, query)
            data.append(text)
        return data

    def get_result(self, query, nresults=20):
        """
        Given query by the user, return the recommended names.
        Parameters:
            query: (type: str) query words from user input, e.g. apple picking
            nresults: number of results to be generated.
        Returns:
            A list of the generated business names by the model.
        """
        data = self._input_data_prep(query, nresults)
        outputs = self.generate(data)
        out_dict = {"queries": [query] * len(outputs), "texts": outputs}
        df_texts = pd.DataFrame(out_dict)
        # only keep the queries with generated texts.
        df_texts = df_texts.dropna(subset=["texts"])
        df_texts = df_texts.drop_duplicates(subset=["texts"])
        logger.info("N requested: %d, unique result generated: %d, query: '%s'"
                    % (nresults, len(df_texts), query))
        try:
            df_texts.to_csv(self.output_file, index=False)
            logger.info("Output saved to: %s" % self.output_file)
        except Exception:
            logger.info("Output not saved to: %s!" % self.output_file)
        return df_texts


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config.toml", type=str, required=True,
        help="Your model tester config file.",
    )
    parser.add_argument(
        "--query", "-q", default=None, type=str,
        required=True, help="Your query string.",
    )
    parser.add_argument(
        "--number-results", "-n", default=2, type=int,
        required=False, help="Your number of results to generate.",
    )
    parser.add_argument(
        "--random-seed", "-rnd", default=None, type=int,
        required=False, help="Re-set your random seed for name generation",
    )

    args = parser.parse_args()
    logger.info("Reading config file: %s" % args.config)
    config = toml.load(args.config)
    if args.random_seed:
        config["random_seed"] = args.random_seed

    query = args.query
    num_results = args.number_results
    t = Inference(config)
    # reset the random seed
    if args.random_seed is not None:
        t.reset_random_seed(args.random_seed)
    t_start = time()
    df = t.get_result(query, num_results)
    logger.info('Output examples:\n%s' % df["texts"].head(20).to_string())

    tdif = time() - t_start
    logger.info("Testing model done!")
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
