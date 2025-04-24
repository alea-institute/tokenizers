import argparse
import glob
from os.path import join

from tokenizers import ByteLevelBPETokenizer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="The files to use as training; accept '**/*.txt' type of patterns \
                          if enclosed in quotes",
)
parser.add_argument(
    "--out",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument("--name", default="bpe-bytelevel", type=str, help="The name of the output vocab files")
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)


# Initialize an empty tokenizer
tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

# And then train with memory optimization
tokenizer.train(
    files,
    vocab_size=10000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>"],
    # Memory optimization parameters
    prune_min_frequency=3,         # Remove pairs with frequency < 3
    prune_word_interval=5000,      # Prune after processing every 5k words
    prune_step_interval=2000,      # Prune after every 2000 merge steps
    prune_keep_percent=0.9,        # Keep only top 90% most frequent pairs
)

# Save the files
tokenizer.save_model(args.out, args.name)

# Restoring model from learned vocab/merges
tokenizer = ByteLevelBPETokenizer(
    join(args.out, "{}-vocab.json".format(args.name)),
    join(args.out, "{}-merges.txt".format(args.name)),
    add_prefix_space=True,
)

# Test encoding
print(tokenizer.encode("Training ByteLevel BPE is very easy").tokens)
