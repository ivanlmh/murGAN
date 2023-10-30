import unittest
import torch
from src.models import Generator, Discriminator

class ModelsTest(unittest.TestCase):

    def test_generator(self):
        generator = Generator()
        self.assertTrue(len(list(generator.parameters())) > 0)
        sample = torch.randn(1, 1, 128, round(44100*10/512 + 0.5)) # 1 sample, 1 channel, 128xsegment_length/hop_length
        output = generator(sample)
        self.assertEqual(output.shape, sample.shape)

    def test_discriminator(self):
        discriminator = Discriminator()
        self.assertTrue(len(list(discriminator.parameters())) > 0)
        sample = torch.randn(1, 1, 128, round(44100*10/512 + 0.5))
        real_fake, domain = discriminator(sample)
        self.assertEqual(real_fake.shape, torch.Size([1, 1]))
        self.assertEqual(domain.shape, torch.Size([1, 1]))
