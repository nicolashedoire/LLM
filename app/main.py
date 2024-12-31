import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import logging

from model import AdvancedLLM, generate_text
from preprocess import preprocess_data
from model import post_process_text

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Variables globales pour stocker le modèle et les données
model = None
vocab = None
preprocessed_data = None

@app.on_event("startup")
async def startup_event():
    global preprocessed_data
    preprocessed_data = preprocess_data("data/divorce_corpus.txt", seq_length=30)
    if preprocessed_data is None:
        logging.error("Failed to preprocess data")
    else:
        logging.info("Données de divorce chargées et prétraitées.")

@app.post("/train")
async def train_model():
    global model, vocab, preprocessed_data
    if preprocessed_data is None:
        raise HTTPException(status_code=500, detail="Erreur: Données non chargées")

    vocab = preprocessed_data['vocab']
    input_sequences = preprocessed_data['input_sequences']
    target_sequences = preprocessed_data['target_sequences']

    vocab_size = len(vocab)
    embedding_dim = 512
    hidden_dim = 512
    num_layers = 4

    model = AdvancedLLM(vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.01)

    num_epochs = 1300
    max_batch_size = 32
    min_batch_size = 16
    curriculum_steps = 3

    dataset = TensorDataset(input_sequences, target_sequences)

    best_loss = float('inf')
    patience = 40
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        current_batch_size = min(max_batch_size, max(min_batch_size, min_batch_size + epoch * curriculum_steps))
        dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)
        
        for batch_input, batch_target in dataloader:
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output.view(-1, vocab_size), batch_target.view(-1))
            
            # Ajout de la régularisation L2
            l2_lambda = 0.0005  # Réduit de 0.001 à 0.0005
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:  # Changé de 10 à 20 pour réduire la fréquence des logs
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Batch Size: {current_batch_size}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    return {"message": "Modèle entraîné avec succès"}

@app.post("/generate")
async def generate(prompt: str = Form(...), temperature: float = Form(0.7), top_p: float = Form(0.9), beam_size: int = Form(5)):
    global model, vocab
    if model is None or vocab is None:
        raise HTTPException(status_code=500, detail="Veuillez d'abord entraîner le modèle")

    start_sequence = [vocab.get(word, vocab['<UNK>']) for word in prompt.split()]
    generated_text = generate_text(model, vocab, start_sequence, max_length=100, temperature=temperature, top_p=top_p, beam_size=beam_size)
    processed_text = post_process_text(generated_text)
    return {"generated_text": processed_text}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Assistant Juridique IA - Divorce</title>
        </head>
        <body>
            <h1>Assistant Juridique IA - Spécialiste du Divorce</h1>
            <form action="/train" method="post">
                <input type="submit" value="Entraîner le Modèle">
            </form>
            <form action="/generate" method="post">
                <input type="text" name="prompt" placeholder="Entrez votre question sur le divorce...">
                <input type="number" name="temperature" step="0.1" min="0.1" max="2" value="0.7">
                <input type="submit" value="Générer une Réponse">
            </form>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)