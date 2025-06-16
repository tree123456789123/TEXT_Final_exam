import torch
from torch.utils.data import DataLoader
from data_loader import MultiModalDataset
from model import CrossAttentionFusion
from train import train_model
from evaluate import evaluate_model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_map = {
        'rescue_volunteering_or_donation_effort': 0,
        'infrastructure_and_utility_damage': 1,
        'affected_individuals': 1,
        'not_humanitarian': 2,
        'other_relevant_information': 2
    }

    train_dataset = MultiModalDataset('reduced_train.csv', 'data_image', label_map)
    test_dataset = MultiModalDataset('reduced_test.csv', 'data_image', label_map)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = CrossAttentionFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        train_loss = train_model(model, train_loader, optimizer, device)

    evaluate_model(model, test_loader, device)
