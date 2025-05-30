import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class RelationNet(nn.Module):
        def __init__(self, n_names, emb_dim, n_num_feats, n_classes):
            super().__init__()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.src_emb = nn.Embedding(n_names, emb_dim)
            self.tgt_emb = nn.Embedding(n_names, emb_dim)

            self.fc = nn.Sequential(
                nn.Linear(2*emb_dim + n_num_feats, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes)
            )
        def forward(self, cat_feats, num_feats):
            src_id, tgt_id = cat_feats[:,0], cat_feats[:,1]
            src_e = self.src_emb(src_id)
            tgt_e = self.tgt_emb(tgt_id)
            x = torch.cat([src_e, tgt_e, num_feats], dim=1)
            return self.fc(x)

def load_model(model_path, device):
    """
    Load the model from the specified path.
    """
    # Load the mappings for object names and labels
    with open('relationship_prediction/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    # Load the model
    name2idx = mappings['name2idx']
    label2idx = mappings['label2idx']
    idx2label = mappings['idx2label']

    model = RelationNet(
        n_names=len(name2idx),
        emb_dim=16,
        n_num_feats=10,
        n_classes=len(label2idx)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, name2idx, label2idx, idx2label

# main function to generate edge relationships
def generate_edge_relationship(object1, object2, model, name2idx, label2idx, idx2label):
    
    # Prepare input data
    src = object1
    tgt = object2
    dist = np.hypot(np.hypot(src.x - tgt.x, src.y - tgt.y), src.z - tgt.z)
    
    # our numerical features + categorical features on top
    input_data = {
        'src_name': src.name,
        'tgt_name': tgt.name,
        'dx': src.x - tgt.x,
        'dy': src.y - tgt.y,
        'dz': src.z - tgt.z,
        'dist': dist,
        'size_x1': src.size_x,
        'size_y1': src.size_y,
        'size_z1': src.size_z,
        'size_x2': tgt.size_x,
        'size_y2': tgt.size_y,
        'size_z2': tgt.size_z
    }

    # we return unknown if we came across objects that were not represented in the training data
    if src.name not in name2idx or tgt.name not in name2idx:
        return "unknown"

    # Convert categorical features to indices
    cat_feats = torch.tensor([
        name2idx[input_data['src_name']],
        name2idx[input_data['tgt_name']]
    ], dtype=torch.int64).unsqueeze(0)  # Add batch dimension

    num_feats = torch.tensor([
        input_data['dx'],
        input_data['dy'],
        input_data['dz'],
        input_data['dist'],
        input_data['size_x1'],
        input_data['size_y1'],
        input_data['size_z1'],
        input_data['size_x2'],
        input_data['size_y2'],
        input_data['size_z2']
    ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    cat_feats = cat_feats.to(model.device)
    num_feats = num_feats.to(model.device)

    # inference on relationship model
    with torch.no_grad():
        logits = model(cat_feats, num_feats)
    
    # Get predicted class and return predicted relationship
    pred_idx = logits.argmax(dim=1).item()
    
    return idx2label[pred_idx]

# cannot be trained in this demo unless compile dataset from 3RScan and 3DSSG
def main():
        
    # load coco labels
    coco_labels_simplified = list(json.load(open('label_mapping/coco_name_to_name_simplified.json')).values())

    # Load JSONL data
    data_file = 'data.jsonl'
    rows = []

    # this file was 
    with open('helper_repos/3RScan/relationship_training_data.jsonl', 'r') as f:
        for line in f:
            sample = json.loads(line)
            src = sample['input']['source']
            tgt = sample['input']['target']
            dist = np.hypot(np.hypot(src['x'] - tgt['x'], src['y'] - tgt['y']), src['z'] - tgt['z'])
            if dist > 100:
                continue
            rows.append({
                'src_name':   src['name'],
                'tgt_name':   tgt['name'],
                'dx':         src['x'] - tgt['x'], # important to keep this order since edge has a direction
                'dy':         src['y'] - tgt['y'],
                'dz':         src['z'] - tgt['z'],
                'dist':       dist,
                'size_x1':    src['size_x'],
                'size_y1':    src['size_y'],
                'size_z1':    src['size_z'],
                'size_x2':    tgt['size_x'],
                'size_y2':    tgt['size_y'],
                'size_z2':    tgt['size_z'],
                'relation':   sample['output']
            })
    df = pd.DataFrame(rows)

    # Create mappings for object names
    all_names = pd.concat([df['src_name'], df['tgt_name']]).unique()
    common_names = set(all_names).intersection(set(coco_labels_simplified))
    print(common_names)
    # Filter rows to only include samples with common names
    df = df[df['src_name'].isin(common_names) & df['tgt_name'].isin(common_names)]
    # Exclude samples with specific relations
    exclude_relations = {
        "same color", "same material", "same texture", "same shape",
        "same state", "same symmetry as", "same as", "brighter than", "darker than",
        "mess", "messier than", "cleaner than", "fuller than", "same object type"
    }
    df = df[~df['relation'].isin(exclude_relations)]
    
    # Replace specific relations
    df['relation'] = df['relation'].replace({'right': 'next to', 'left': 'next to'})

    # Show statistics of the types of relationships in df["relation"]
    relation_counts = df['relation'].value_counts()
    print("Relationship statistics:")
    print(relation_counts)

    # Create a dictionary that maps each relation to the indices in the dataframe
    # relation_to_indices = {relation: df.index[df['relation'] == relation].tolist() for relation in df['relation'].unique()}
    print("Relation to indices mapping created.")

    name2idx = {name: i for i, name in enumerate(common_names)}
    n_names = len(name2idx)

    # Map names to indices
    df['src_id'] = df['src_name'].map(name2idx)
    df['tgt_id'] = df['tgt_name'].map(name2idx)

    # Create mapping for relation labels
    all_rels = df['relation'].unique()
    label2idx = {rel: i for i, rel in enumerate(all_rels)}
    idx2label = {i: rel for rel, i in label2idx.items()}
    df['label'] = df['relation'].map(label2idx)
    n_classes = len(all_rels)
    print(f"Found {n_classes} distinct relations.")

    # Numeric features
    num_feats = [
        'dx', 'dy', 'dz', 'dist',
        'size_x1', 'size_y1', 'size_z1',
        'size_x2', 'size_y2', 'size_z2'
    ]
    X_num = df[num_feats].values.astype(np.float32)
    X_cat = df[['src_id','tgt_id']].values.astype(np.int64)
    y = df['label'].values.astype(np.int64)

    X_num_tr, X_num_val, X_cat_tr, X_cat_val, y_tr, y_val = train_test_split(
        X_num, X_cat, y, test_size=0.9, random_state=41, stratify=y
    )

    batch_size = 1000

    train_ds = TensorDataset(
        torch.from_numpy(X_cat_tr), 
        torch.from_numpy(X_num_tr), 
        torch.from_numpy(y_tr)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_cat_val), 
        torch.from_numpy(X_num_val), 
        torch.from_numpy(y_val)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RelationNet(n_names, emb_dim=16, n_num_feats=len(num_feats), n_classes=n_classes).to(device)

    # standard training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 100

    loss_progression = []
    confusion_matrix = torch.zeros(n_classes, n_classes)
    for epoch in range(1, n_epochs+1):
        model.train()
        total_loss = 0.0
        for cat_batch, num_batch, lbl_batch in train_loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            optimizer.zero_grad()
            outputs = model(cat_batch, num_batch)
            loss = criterion(outputs, lbl_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * lbl_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for cat_batch, num_batch, lbl_batch in val_loader:
                cat_batch = cat_batch.to(device)
                num_batch = num_batch.to(device)
                lbl_batch = lbl_batch.to(device)
                logits = model(cat_batch, num_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == lbl_batch).sum().item()
                for t, p in zip(lbl_batch.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        val_acc = correct / len(val_loader.dataset)
        loss_progression.append(avg_loss)
        

        print(f"Epoch {epoch}/{n_epochs}  Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f}")


    # we want to normalize the confusion matrix since different labels appear with different frequencies
    confusion_matrix = confusion_matrix / confusion_matrix.sum(0, keepdim=True)
    confusion_matrix = torch.round(confusion_matrix * 100) / 100 
    
    # Plot confusion matrix for thesis
    plt.figure(figsize=(10, 8))
    conf_matrix_np = confusion_matrix.numpy()
    sns.heatmap(conf_matrix_np, annot=True, fmt='g', cmap='Blues', 
                xticklabels=idx2label.values(), yticklabels=idx2label.values(), 
                annot_kws={"size": 8})  # Reduce annotation text size
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Ground Truth', fontsize=10)
    plt.title('Confusion Matrix', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)  
    plt.yticks(rotation=45, ha='right', fontsize=8)  
    plt.tight_layout()  # Adjust layout to make room for x-axis label
    plt.savefig('dataset/confusion_matrix.png')

    # Plot loss progression for thesis 
    plt.figure(figsize=(10, 5))
    plt.plot(loss_progression, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Progression')
    plt.legend()
    plt.savefig('dataset/loss_progression.png')

    torch.save(model.state_dict(), 'dataset/relation_model.pth')
    with open('dataset/mappings.pkl', 'wb') as f:
        pickle.dump({'name2idx': name2idx, 'label2idx': label2idx, 'idx2label': idx2label}, f)

    print("Training complete. Model saved to 'relation_model.pth' and mappings to 'mappings.pkl'.")

if __name__ == "__main__":
    main()
