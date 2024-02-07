import os
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def make_dir(filefullname):
    """
    Given the full path of the output name, if the folder doesn't exist,
    then create the folder.
    """
    outfold = "/".join(filefullname.split("/")[0:-1])
    if outfold and (not os.path.exists(outfold)):
        os.makedirs(outfold)
        logger.info("Making dir: %s!" % outfold)
