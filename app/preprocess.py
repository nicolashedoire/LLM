import torch
import logging
import spacy
from collections import Counter

logging.basicConfig(level=logging.INFO)

nlp = spacy.load("fr_core_news_sm")

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()
    logging.info(f"Loaded {len(data)} lines from {filename}")
    return data

def normalize_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

def build_vocab(sentences, max_vocab_size=10000):
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence.split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    
    logging.info(f"Built vocabulary with {len(vocab)} words")
    return vocab

def encode_sentences(sentences, vocab):
    encoded = [[vocab.get(word, vocab['<UNK>']) for word in sentence.split()] for sentence in sentences]
    logging.info(f"Encoded {len(encoded)} sentences")
    return encoded

def create_sequences(encoded_sentences, seq_length, vocab):
    sequences = []
    for sentence in encoded_sentences:
        if len(sentence) >= 2:
            padded_sentence = sentence + [vocab['<PAD>']] * (seq_length + 1 - len(sentence))
            for i in range(len(padded_sentence) - seq_length):
                input_seq = padded_sentence[i:i+seq_length]
                target_seq = padded_sentence[i+1:i+seq_length+1]
                sequences.append((input_seq, target_seq))
    logging.info(f"Created {len(sequences)} sequences")
    return sequences

def sort_sequences_by_length(sequences):
    return sorted(sequences, key=lambda x: len(x[0]))

def preprocess_data(filename, seq_length=30):
    sentences = load_data(filename)
    normalized_sentences = [normalize_text(sentence) for sentence in sentences]
    vocab = build_vocab(normalized_sentences)
    encoded_sentences = encode_sentences(normalized_sentences, vocab)
    sequences = create_sequences(encoded_sentences, seq_length, vocab)
    sequences = sort_sequences_by_length(sequences)

    if not sequences:
        logging.error("No sequences created. Check your data and seq_length.")
        return None

    input_sequences = torch.tensor([seq[0] for seq in sequences], dtype=torch.long)
    target_sequences = torch.tensor([seq[1] for seq in sequences], dtype=torch.long)

    logging.info(f"Input sequences shape: {input_sequences.shape}")
    logging.info(f"Target sequences shape: {target_sequences.shape}")

    return {
        'vocab': vocab,
        'input_sequences': input_sequences,
        'target_sequences': target_sequences
    }