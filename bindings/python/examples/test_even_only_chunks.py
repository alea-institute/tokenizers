import os
import tempfile
from tokenizers import Tokenizer, normalizers, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import RandomChunkSplit

# Create a simple test text
test_text = """
This is a test of the even_only parameter for RandomChunkSplit.
We want to train a small BPE tokenizer with this feature
to verify that it works correctly.
"""

# Create a temporary file with test text
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    f.write(test_text)
    temp_file = f.name

try:
    # Initialize tokenizers with different settings for comparison
    tokenizer_standard = Tokenizer(BPE())
    tokenizer_standard.pre_tokenizer = RandomChunkSplit(min_length=2, max_length=6)
    
    tokenizer_even = Tokenizer(BPE())
    tokenizer_even.pre_tokenizer = RandomChunkSplit(min_length=2, max_length=6).with_even_only(True)
    
    # Check pre-tokenizer parameters
    # This should show adjusted min/max lengths to make them even
    print(f"Standard pre-tokenizer: min_length={tokenizer_standard.pre_tokenizer.min_length}, max_length={tokenizer_standard.pre_tokenizer.max_length}")
    print(f"Even-only pre-tokenizer: min_length={tokenizer_even.pre_tokenizer.min_length}, max_length={tokenizer_even.pre_tokenizer.max_length}")
    print(f"Even-only flag set: {tokenizer_even.pre_tokenizer.even_only}")
    
    # Train both tokenizers
    trainer = trainers.BpeTrainer(
        vocab_size=50,
        min_frequency=1,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    print("\nTraining standard tokenizer...")
    tokenizer_standard.train([temp_file], trainer)
    
    print("\nTraining even-only tokenizer...")
    tokenizer_even.train([temp_file], trainer)
    
    # Test tokenization
    test_sentence = "This is a test sentence for our tokenizers."
    print("\nStandard tokenizer results:")
    encoded_standard = tokenizer_standard.encode(test_sentence)
    print(f"Tokens: {encoded_standard.tokens}")
    
    print("\nEven-only tokenizer results:")
    encoded_even = tokenizer_even.encode(test_sentence)
    print(f"Tokens: {encoded_even.tokens}")
    
    # Directly test pre-tokenization to see chunk lengths
    print("\nPre-tokenization comparison:")
    print("Standard pre-tokenization chunks:")
    chunks_standard = tokenizer_standard.pre_tokenizer.pre_tokenize_str(test_sentence)
    for chunk, pos in chunks_standard:
        print(f"'{chunk}' (len={len(chunk)}) at {pos}")
    
    print("\nEven-only pre-tokenization chunks:")
    chunks_even = tokenizer_even.pre_tokenizer.pre_tokenize_str(test_sentence)
    for chunk, pos in chunks_even:
        print(f"'{chunk}' (len={len(chunk)}) at {pos}")
        # Verify that all chunk lengths are even numbers (except possibly the last chunk)
        if pos[1] < len(test_sentence) - 1:  # Not the last chunk
            assert len(chunk) % 2 == 0, f"Chunk '{chunk}' has odd length {len(chunk)}"
    
    print("\nAll tests passed! The even_only parameter is working correctly.")
    
finally:
    # Clean up
    os.unlink(temp_file)