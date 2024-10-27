import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 检查设备是否可用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 数据路径
train_dir = './datasets/train'
val_dir = './datasets/val'
test_dir = './datasets/test'
csv_path = './test_labels.csv'

# 超参数设置
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 数据增强与预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载训练和验证集
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 定义模型
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 二分类
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练和验证函数
def train_and_validate():
    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f'Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%')

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print('Training complete. Best validation accuracy: {:.2f}%'.format(best_acc))

# 加载测试数据并进行预测
def load_test_data_from_csv(csv_path, test_dir):
    data = pd.read_csv(csv_path)
    images = []
    ids = data['id'].astype(str)

    for img_id in ids:
        img_path = os.path.join(test_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img = data_transforms['test'](img)
            images.append((img, img_id))
        else:
            print(f"Warning: {img_path} not found!")

    return images

def predict(images):
    model.eval()
    results = []

    with torch.no_grad():
        for img, img_id in images:
            img = img.unsqueeze(0).to(device)
            output = model(img)
            _, predicted = torch.max(output, 1)
            label = 'dog' if predicted.item() == 1 else 'cat'
            results.append((img_id, label))

    return results


# 主函数
if __name__ == "__main__":
    train_and_validate()

    # 加载测试集并进行预测
    test_images = load_test_data_from_csv(csv_path, test_dir)
    predictions = predict(test_images)

    # 保存预测结果为 CSV
    submission = pd.DataFrame(predictions, columns=['id', 'predicted_label'])
    submission.to_csv('submission.csv', index=False)

    print('Predictions saved to submission.csv')
