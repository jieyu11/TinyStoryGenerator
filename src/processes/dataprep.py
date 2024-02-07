import argparse
import pandas as pd
from datetime import timedelta
from time import time
import toml
from sklearn.model_selection import train_test_split
import spacy
import random
import os

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPrep:
    """
    Class to prepare text data for GPT-2 (and its variants) model training.
    """
    def __init__(self, config):
        """
        Initialization of the DataPrep class.
        Parameters:
            config: dict of configuration parameters, the keys are listed.
            - input_name: (str) input raw file name.
            - split_random_seed, (int, optional, default 12345678), random seed
            to split the data into training and testing.
            - train_fraction: (int, optional, default 0.99) the fraction for
            training data.
            - starter: (str, optional, default "<|startoftext|>") the starting string
            to be placed at the beginning of a training text.
            - ender: (str, optional, "<|endoftext|>") the ending of a training text.
            - outname: (str, optional, default "data_prepared.csv") output
            file name to save the results.
        """
        self.input_name = config.get("input_name", None)
        self.split_rndstate = config.get("split_random_seed", -1)
        self.train_fraction = config.get("train_fraction", 0.95)
        self.ndata = config.get("ndata", -1)
        self.starter = config.get("ender", "<|startoftext|>")
        self.ender = config.get("ender", "<|endoftext|>")
        self.outname = config.get("outname", "data_prepared.csv")
        # check if the output folder exists, if not, create one!
        outfold = "/".join(self.outname.split("/")[0:-1])
        if outfold and (not os.path.exists(outfold)):
            os.makedirs(outfold)
            logger.info("Making dir: %s for output!" % outfold)

    def _read_input(self):
        """
        Read from the input file by given file name and column names.
        Parameters:
            input_name: (str) input file name: input.txt
        """
        assert os.path.exists(self.input_name), "%s doesn't exist!" % self.input_name
        logger.info("Reading input file: %s" % self.input_name)
        
        # Read the text file line by line until <|endoftext|>
        current_text = ""
        texts = list()

        with open(self.input_name) as file:
            while line := file.readline():
                if self.ender in line:
                    out = self.starter + " " + current_text.replace("\n", "") + self.ender
                    while "  " in out: out = out.replace("  ", " ")
                    texts.append(out)
                    current_text = ""
                    if self.ndata > 0 and len(texts) >= self.ndata:
                        break
                elif len(line) > 0:
                    current_text += line + " "
        
        logger.info("Number of texts processed: %d" % len(texts))
        
        df = pd.DataFrame({"texts": texts})
        return df

    def prepare_data(self):
        """
        Prepare data with format like:
            <|startoftext|> texts texts texts. <|endoftext|>
        """
        df = self._read_input()
        if self.split_rndstate >= 0:
            self.split_data(df)
        else:
            df.to_csv(self.outname, index=False)
            logger.info("N data: %d in %s." % (len(df), self.outname))

    def split_data(self, df):
        # split the output file into training and testing.
        df_train, df_test = train_test_split(
            df, train_size=self.train_fraction, shuffle=True,
            random_state=self.split_rndstate)

        # save the output file for training and testing.
        trainname = self.outname.replace(".csv", "_train.csv")
        df_train.to_csv(trainname, index=False)
        logger.info("N train data: %d in %s." % (len(df_train), trainname))

        testname = self.outname.replace(".csv", "_test.csv")
        df_test.to_csv(testname, index=False)
        logger.info("N test data: %d in %s." % (len(df_test), testname))


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.toml", type=str,
                        required=False, help="Config file.",)
    args = parser.parse_args()
    logger.info("Reading config file: %s" % args.config)
    config = toml.load(args.config)
    dp = DataPrep(config)
    dp.prepare_data()

    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()
