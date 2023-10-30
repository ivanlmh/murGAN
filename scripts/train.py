import torch
from torch import optim
from torch.nn import functional as F
from src.models import Generator, Discriminator
from src.data_loader import create_data_loader


# Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/murga_classifier")


# Hyperparameters
lr = 0.0001
n_epochs = 100
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create data loaders
murga_data_loader = create_data_loader(
    root_dir="data/murga", batch_size=batch_size, shuffle=True, num_workers=0
)
classical_data_loader = create_data_loader(
    root_dir="data/classic", batch_size=batch_size, shuffle=True, num_workers=0
)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Losses
adversarial_loss = torch.nn.BCEWithLogitsLoss()
domain_loss = torch.nn.CrossEntropyLoss()

# Dynamic weighting for domain translation loss
alpha = 0.01  # initial weight
alpha_increment = 0.01  # the amount by which alpha is increased after each epoch
alpha_max = 0.5  # maximum weight

# Training loop
for epoch in range(n_epochs):
    for i, (murga_samples, classical_samples) in enumerate(
        zip(murga_data_loader, classical_data_loader)
    ):
        # Move to device
        murga_samples = murga_samples.to(device)
        classical_samples = classical_samples.to(device)

        # Create labels
        real = torch.ones((murga_samples.size(0), 1)).to(device)
        fake = torch.zeros((murga_samples.size(0), 1)).to(device)
        murga_domain = torch.zeros((murga_samples.size(0), 1)).long().to(device)
        classical_domain = torch.ones((classical_samples.size(0), 1)).long().to(device)

        # -----------------
        #  Train Discriminator
        # -----------------
        d_optimizer.zero_grad()

        # Adversarial loss
        # Train on real samples
        real_mel_logits, real_domain_logits = discriminator(murga_samples)
        d_real_loss = adversarial_loss(real_mel_logits, real)

        # Train on fake samples
        fake_murga_samples = generator(classical_samples)
        fake_mel_logits, _ = discriminator(fake_murga_samples)
        d_fake_loss = adversarial_loss(fake_mel_logits, fake)

        # Domain loss
        d_domain_loss = domain_loss(real_domain_logits, murga_domain)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + d_domain_loss
        d_loss.backward()
        d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        g_optimizer.zero_grad()

        # Adversarial loss
        fake_murga_samples = generator(classical_samples)
        fake_mel_logits, fake_domain_logits = discriminator(fake_murga_samples)
        g_fake_loss = adversarial_loss(fake_mel_logits, real)

        # Domain transfer loss
        g_domain_loss = domain_loss(fake_domain_logits, classical_domain)

        # Total generator loss
        g_loss = g_fake_loss * (1 - alpha) + g_domain_loss * alpha
        alpha = min(alpha + alpha_increment, alpha_max)
        g_loss.backward()
        g_optimizer.step()

        # Print losses occasionally and print to tensorboard
        if i % 10 == 0:
            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(murga_data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )
            # Tensorboard
            writer.add_scalar(
                "Loss/D", d_loss.item(), global_step=epoch * len(murga_data_loader) + i
            )
            writer.add_scalar(
                "Loss/G", g_loss.item(), global_step=epoch * len(murga_data_loader) + i
            )
            writer.add_scalar(
                "Loss/D_real",
                d_real_loss.item(),
                global_step=epoch * len(murga_data_loader) + i,
            )
            writer.add_scalar(
                "Loss/D_fake",
                d_fake_loss.item(),
                global_step=epoch * len(murga_data_loader) + i,
            )
            writer.add_scalar(
                "Loss/D_domain",
                d_domain_loss.item(),
                global_step=epoch * len(murga_data_loader) + i,
            )
            writer.add_scalar(
                "Loss/G_fake",
                g_fake_loss.item(),
                global_step=epoch * len(murga_data_loader) + i,
            )
            writer.add_scalar(
                "Loss/G_domain",
                g_domain_loss.item(),
                global_step=epoch * len(murga_data_loader) + i,
            )
            writer.add_scalar(
                "Alpha", alpha, global_step=epoch * len(murga_data_loader) + i
            )

    # Save model checkpoints
    torch.save(generator.state_dict(), "saved_models/generator.pth")
    torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")

writer.close()
