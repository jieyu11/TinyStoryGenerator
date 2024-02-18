# Tiny Story Generator
This project aims to fine-tune a relatively small language model (e.g., GPT
small (117M), medium (345M), large (774M), and extra-large (1.5B) respectively)
on short stories to produce outputs of similar quality.

Researchers at Microsoft have already achieved this, as documented in their
paper [arxiv:2305.07759](https://arxiv.org/abs/2305.07759).

It's important to note that the technique employed in this package is not
confined to this specific use case. Instead, it can be applied to any text
generation fine-tuning with models available on HuggingFace.

## Installation
The package has been tested with `Python 3.9.6` and the following modules along
with their corresponding versions:
```
torch                        2.1.1
torchvision                  0.16.1
pandas                       2.1.3
toml                         0.10.2
transformers                 4.35.2
peft                         0.7.0
```

## Data
The data is retrieved from Huggingface with the dataset named TinyStories. An
example of a tiny story is as follows:

```
<|startoftext|> Once upon a time, there was a little boy named Tim. Tim was
a happy boy who liked to play with his toys. One day, Tim saw a pretty m irror
on the wall. He wanted to look at himself, but he was too small to see. He tried
to jump, but he could not manage to see his face. Tim was sad that he could not
see himself in the mirror. He thought of a plan to fix the problem. Tim went to
his room and found his big toy box. He pushed the b ox across the floor to the
mirror. It was hard work, but Tim did not give up. Finally, Tim managed to get
the toy box close to the mirror. He climbed on top of the box and looked into
the mirror. Tim saw his happy face and smiled. He was proud of himself for
solving the problem. From that day on, Tim knew that he could do anything if he
tried his best. <|endoftext|>
```

### Data Processing
In this package, data is processed into a `csv` file with at least one column
named `text`. Each row in this column is a data point, which is a tiny story in
our case. Each data point starts with `<|startoftext|>` and ends with
`<|endoftext|>`.

To convert the original `TinyStory` data from HuggingFace to the above-mentioned format, follow these steps:
```
python3 src/processes/dataprep.py -c config/dataprep.toml
```

The following parameters are defined in the config file:

```
ndata = 1000
train_fraction = 0.95
split_random_seed = 12345678
input_name = "data/original/TinyStoriesV2-GPT4-train.txt"
starter = "<|startoftext|>"
ender = "<|endoftext|>"
outname = "data/processed/stories_1000.csv"
```

## Model Finetuning
To fine-tune the model, simply run the following code:
```
python3 src/modeling/finetune.py -c config/model_finetune.toml 
```

In the config file, the following parameters are set:
```
input_file = "data/processed/stories_1000_test.csv"
lora_peft = true
model_folder = "models/finetuned/peft_full"
save_per_epoch = false
max_seq_len = -1
model_name = "distilgpt2"
device = "cpu"
```

## Model Inference

Once the model is fine-tuned, it can be used for inference with the following command:
```
python3 src/modeling/inference.py -c config/model_inference.toml -q "Once \
upon a time, in an ancient house, there lived a girl named Lily. She loved to \
decorate her room with pretty things. One day, she found a box" -n 1 -rnd 876543
```

The parameters set in the config are:
```

output_file = "output/test_gpt-xl_t125.csv"
# -------------------
# ---- The model ----
# -------------------

lora_peft = true
model_folder = "models/finetuned/peft_full"
model_name = "distilgpt2"
quantized = true
device = "cpu"
starter = "<|startoftext|>"
ender = "<|endoftext|>"
# set up the hyper-parameters for text generation.
[hyper-parameters]
  max_new_tokens = 1096
  temperature = 1.25
  top_k = 250
  top_p =1.0
  repetition_penalty = 1.0
  length_penalty = 1
  bad_words = ["Damn", "Shit", "Fuck"]
```

One example of the output can be found below:

```
Once upon a time, in an ancient house, there lived a girl named Lily. She loved
to decorate her room with pretty things. One day, she found a box and opened it.
Then, she looked inside and was heartened. All items in the box were perfectly
wrapped and packaged, perfect for the holidays! She opened two more boxes: The
second contained articles each intended to fit in the first. After these lovely
presents from her friend were opened, Lily declared: "It's definitely Christmas
about now." The Cat has told stories full of elves and gold but now her work as
a professional illustrator has an exciting new star. A little penguin said in
the following days what you already know: The Snow Cat™ announces: "It's truly
and truly the Most Holy Holidays weekend of our entire lives!" She has chosen a
new holiday: Christmas. Everyone who sees that penguin will have a favorite
Christmas animal! Follow Snowcat™ on Twitter (@Cat_Itsnot) to receive a stunning
Penguin Snow Box. • Collect snow.

- Wait what?
- Meow! If you're interested, answer on the comments on where I find a map!
Where are the penguins hiding-forever in this penguin cake?
```