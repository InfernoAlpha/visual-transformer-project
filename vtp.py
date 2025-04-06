import torch
from torch import nn
from torch.nn import functional as f
from dataclasses import dataclass
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from pathlib import Path
import math
from tqdm import tqdm
import torch.optim as optim

transform1 = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.TrivialAugmentWide(num_magnitude_bins=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform2 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data_path = Path(r"C:\Users\chara\Desktop\Desktop\vs code\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train/")
test_data_path = Path(r"C:\Users\chara\Desktop\Desktop\vs code\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid/")

train_data = datasets.ImageFolder(root=train_data_path,
                                  transform=transform1,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_data_path,transform=transform2)

train_dataloder = DataLoader(dataset=train_data,batch_size=4,num_workers=0,shuffle=True)
test_dataloder = DataLoader(dataset=test_data,batch_size=4,num_workers=0,shuffle=False)

@dataclass
class config:
    num_channels:int = 3
    embed_dim:int = 768
    img_size:int = 224
    patch_size:int = 16
    num_attention_heads:int = 12
    attention_dropout:float = 0.4
    intermidiate_size:int = 3072
    layer_norm_eps: float = 1e-6

class patch_embed(nn.Module):
    def __init__(self,config:config):
        super().__init__()
        self.num_channels = config.num_channels
        self.embed_dim = config.embed_dim
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patch = (self.img_size//self.patch_size)**2
        self.num_position = self.num_patch
        self.pos_embed = nn.Embedding(self.num_position,self.embed_dim)
        self.register_buffer("position_ids",
                            torch.arange(self.num_position).expand((1,-1)),
                            persistent=False,
                            )
        
    def forward(self,pixel_values:torch.FloatTensor) -> torch.Tensor:
        B,C,H,W = pixel_values.shape

        patch_embed = self.patch_embedding(pixel_values)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = patch_embed.flatten(2,-1)
        embeddings = embeddings.transpose(1,2)
        embeddings = embeddings + self.pos_embed(self.position_ids)
        embeddings = torch.cat((self.cls_token.expand(B, 1, -1), embeddings), 1)
        return embeddings

class selfattn(nn.Module):
    def __init__(self,config:config):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.embed_dim = self.config.embed_dim
        self.dropout = self.config.attention_dropout
        
        self.key_proj = nn.Linear(in_features=self.embed_dim,out_features=self.embed_dim)
        self.query_proj = nn.Linear(in_features=self.embed_dim,out_features=self.embed_dim)
        self.value_proj = nn.Linear(in_features=self.embed_dim,out_features=self.embed_dim)
        self.out_proj = nn.Linear(in_features=self.embed_dim,out_features=self.embed_dim)
    
    def forward(self,hiddenstates):
        B , T , C = hiddenstates.shape

        q_states = self.query_proj(hiddenstates)
        k_states = self.key_proj(hiddenstates)
        v_states = self.key_proj(hiddenstates)

        q_states = q_states.view(B ,T ,self.num_heads,C // self.num_heads).transpose(1,2)
        k_states = k_states.view(B ,T ,self.num_heads,C // self.num_heads).transpose(1,2)
        v_states = v_states.view(B ,T ,self.num_heads,C // self.num_heads).transpose(1,2)

        attn_weights = (q_states @ k_states.transpose(-2,-1)) * (1.0 / math.sqrt(k_states.size(-1)))
        attn_weights = f.softmax(attn_weights,dim=-1).to(q_states.dtype)
        attn_weights = f.dropout(attn_weights,p = self.dropout,training= self.training)
        attn_out = attn_weights @ v_states
        attn_out = attn_out.transpose(1,2)
        attn_out = attn_out.reshape(B,T,C).contiguous()
        attn_out = self.out_proj(attn_out)
        return attn_out

class MultiLevelPerceptron(nn.Module):
    def __init__(self,config:config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(in_features=self.config.embed_dim,out_features=self.config.intermidiate_size)
        self.fc2 = nn.Linear(in_features=self.config.intermidiate_size,out_features=self.config.embed_dim)

    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class encoder_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        self.self_attn = selfattn(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps=self.config.layer_norm_eps)
        self.mlp = MultiLevelPerceptron(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,eps=self.config.layer_norm_eps)

    def forward(self,hidden_state):
        residual = hidden_state
        hidden_state = self.layer_norm1(hidden_state)
        hidden_state = self.self_attn(hidden_state)
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state
        
        return hidden_state

class vision_transformer(nn.Module):
    def __init__(self,config:config):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.embed_dim = self.config.embed_dim
        self.dropout = self.config.attention_dropout
        self.patch_embed = patch_embed(config)
        self.mlp = MultiLevelPerceptron(config)
        self.fc_out = nn.Linear(self.embed_dim, 38)
        self.encoder_layers = nn.ModuleList([
            encoder_layer() for _ in range(self.num_heads)
        ])

    def forward(self,x):
        x = self.patch_embed(x)
        for encoder in self.encoder_layers:
            x = encoder(x)
        cls_output = x[:,0]
        logits = self.fc_out(cls_output)
        return logits
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20

    model = vision_transformer(config).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-4,weight_decay=0.01)

    for epoch in range(epochs):
        loop = tqdm(enumerate(train_dataloder),total=len(train_dataloder))
        for batch_idx,(data,targets) in loop:
            data = data.to(device)
            target = targets.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred,target)
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch+1}]')
            loop.set_postfix(loss = loss.item())

    def check_accuracy(loader,model):
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x,y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                _, pred = logits.max(1)
                num_correct += (pred == y).sum()
                num_samples += pred.size(0)
        model.train()
        return num_correct/num_samples

    print(check_accuracy(test_dataloder, model))

    save_path = Path("vit_model_MP2.pth")
    torch.save(model.state_dict(),save_path)