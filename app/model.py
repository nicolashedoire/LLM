import torch
import re
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        feed_forward_out = self.feed_forward(x)
        return self.norm2(x + self.dropout(feed_forward_out))

class AdvancedLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=4):
        super(AdvancedLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.Embedding(1000, embedding_dim)
        self.layers = nn.ModuleList([AttentionLayer(hidden_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        x = self.dropout(self.embedding(x) + self.position_encoding(positions))
        
        for layer in self.layers:
            x = layer(x)
        
        return self.fc(x)
    
def post_process_text(text):
    # Capitaliser la première lettre de chaque phrase
    text = '. '.join(sentence.capitalize() for sentence in text.split('. '))
    
    # Corriger les espaces avant la ponctuation
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    
    # Ajouter des espaces après la ponctuation si nécessaire
    text = re.sub(r'([,.!?])([^\s])', r'\1 \2', text)
    
    # Remplacer les répétitions de mots
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    
    # Assurer qu'il y a un point final
    if not text.endswith('.'):
        text += '.'
    
    return text

def generate_text(model, vocab, start_seq, max_length=100, temperature=0.7, top_k=0, top_p=0.9, beam_size=5):
    model.eval()
    inv_vocab = {i: word for word, i in vocab.items()}
    
    # Initialisation du beam
    beams = [(start_seq, 0)]  # (sequence, score)
    complete_beams = []

    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == vocab['<PAD>']:
                complete_beams.append((seq, score))
                continue

            with torch.no_grad():
                output = model(torch.tensor([seq]))
                output = output[0, -1, :] / temperature
                
                # Appliquer top-k si spécifié
                if top_k > 0:
                    indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
                    output[indices_to_remove] = float('-inf')
                
                # Appliquer top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(output, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(dim=0, index=sorted_indices, src=sorted_indices_to_remove)
                    output[indices_to_remove] = float('-inf')
                
                probs = F.softmax(output, dim=-1)
                
                # Sélectionner les top_k candidats
                top_probs, top_indices = torch.topk(probs, k=beam_size)
                
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score - torch.log(prob).item()
                    new_beams.append((new_seq, new_score))
        
        # Garder les beam_size meilleures séquences
        beams = sorted(new_beams, key=lambda x: x[1])[:beam_size]
        
        if len(complete_beams) >= beam_size:
            break
    
    # Choisir la meilleure séquence complète, ou la meilleure incomplète si aucune n'est complète
    if complete_beams:
        best_seq, _ = min(complete_beams, key=lambda x: x[1])
    else:
        best_seq, _ = min(beams, key=lambda x: x[1])
    
    return ' '.join(inv_vocab[token] for token in best_seq if token not in [vocab['<PAD>'], vocab['<UNK>']])