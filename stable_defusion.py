weights_file = 'C:\\Users\\bisho\\Downloads\\sd-v1-4.ckpt'

from convoluted.nn.utils import fake_torch_load_zipped, get_child
from convoluted.nn.layers import Conv2D
from convoluted.nn.functions import swish
import numpy as np

dat = fake_torch_load_zipped(open(weights_file, "rb"), load_weights=False)

class Normalize:
    def __init__(self, in_channels, num_groups=32):
        scale = 2 / np.sqrt(in_channels * 3 * 3)

        self.weights = np.random.normal(scale=scale, size=(in_channels, in_channels, *(3, 3)))
        self.bias = np.zeros(shape=(in_channels, 1))        

    def foward(self, x):
        pass


class AttnBlock:
    def __init__(self, in_channels):
        self.norm = Normalize(in_channels)
        self.q = Conv2D(in_channels, in_channels, 1)
        self.k = Conv2D(in_channels, in_channels, 1)
        self.v = Conv2D(in_channels, in_channels, 1)
        self.proj_out = Conv2D(in_channels, in_channels, 1)

class ResnetBlock:
    def __init__(self, in_channels, out_channels=None):
        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv2D(in_channels, out_channels, 3)
        self.norm1 = Normalize(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, 3)
        if in_channels != out_channels:
            self.nin_shortcut = Conv2D(in_channels, out_channels, 1)

class Decoder:
    def __init__(self):
        sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
        self.conv_in = Conv2D(4, 512, 3)

        arr = []
        for i,s in enumerate(sz):
            x = {}
            x['upsample'] = {"conv": Conv2D(s[1], s[1], 3, stride=2, padding=(0, 1, 0, 1))}
            x["block"] = [ResnetBlock(s[1], s[0]), 
            ResnetBlock(s[0], s[0]), 
            ResnetBlock(s[0], s[0])]
            arr.append(x)
        self.up = arr
            
        block_in = 512
        self.mid = {
            "block_1": ResnetBlock(block_in, block_in),
            "attn_1": AttnBlock(block_in),
            "block_2": ResnetBlock(block_in, block_in)
        }

        self.norm_out = Normalize(128)
        self.conv_out = Conv2D(128, 3, 3)

    def foward(self, x):
        x = self.conv_in(x)

        x = x.mid['block_1'](x)
        x = x.mid['attn_1'](x)
        x = x.mid['block_2'](x)

        for l in self.up:
            if 'upsample' in l: x = l['upsample'](x)
            for b in l["block"]: x = b.foward(x)

        return self.conv_out(swish(self.norm_out(x)))

class Encoder:
    def __init__(self):
        sz = [(128, 128), (128, 256), (256, 512), (512, 512)]

        # if decode:
        #     sz = [(x[1], x[0]) for x in sz[::-1]]

        self.conv_in = Conv2D(3, 128, 3)

        arr = []
        for i,s in enumerate(sz):
            arr.append({"block": 
                [ResnetBlock(s[0], s[1]),
                ResnetBlock(s[1], s[1])
                ]})
            if i != 3: arr[-1]['downsample'] = {"conv": Conv2D(s[1], s[1], 3, stride=2, padding=(0, 1, 0, 1))}
        self.down = arr

        # self.down = []

        # if decode:
        #     block_in = 128
        # else:
        #     block_in = 512

        block_in = 512

        self.mid = {
            "block_1": ResnetBlock(block_in, block_in),
            "attn_1": AttnBlock(block_in),
            "block_2": ResnetBlock(block_in, block_in)
        }

        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2D(block_in, 8, 3)

    def foward(self, x):
        x = self.conv_in(x)
        for l in self.down:
            for b in l["block"]: x = b(x)
            if 'downsample' in l: x = l['downsample'](x)
        
        x = x.mid['block_1'](x)
        x = x.mid['attn_1'](x)
        x = x.mid['block_2'](x)

        return self.conv_out(swish(self.norm_out(x)))

class AutoencoderKL:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_quav = Conv2D(8, 8, 1)
        self.post_quant_quav = Conv2D(4, 4, 1)

    def foward(self, x):
        latent = self.encoder.foward(x)
        latent = self.quant_quav(latent)
        latent = latent[:, 0:4]
        latent = self.post_quant_quav(latent)
        return self.decoder.foward(latent)

class Stablediffusion:
    def __init__(self):
        self.first_stage_model = AutoencoderKL()

    def foward(self, x):
        return self.first_stage_model.foward(x)

model = Stablediffusion()

for k,v in dat['state_dict'].items():
    try:
        w = get_child(model, k)
    except (AttributeError, KeyError, IndexError):
        w = None
    print(f"{str(v.shape):30s}", w, k)
    if w is not None:
        assert w.shape == v.shape

   
IMG = "grid-0006.png"
from PIL import Image
img = np.array(Image.open(IMG)).reshape((1, 512, 768))
print(img.shape)